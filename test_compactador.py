import unittest

import compactador as c


def user(text, cache=False):
    b = {"type": "text", "text": text}
    if cache:
        b["cache_control"] = {"type": "ephemeral"}
    return {"role": "user", "content": [b]}


def asst_tool(i, text="x" * 2000):
    return {"role": "assistant", "content": [{"type": "text", "text": text},
                                             {"type": "tool_use", "id": f"t{i}", "name": "Read",
                                              "input": {"file_path": f"/f{i}.py"}}]}


def result(i, body="r" * 3500):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": body}]}


def conv(turns=10):
    msgs = [{"role": "user", "content": [{"type": "text", "text": "<system-reminder>CLAUDE.md</system-reminder>"},
                                         {"type": "text", "text": "hola"}]}]
    for i in range(turns):
        msgs += [asst_tool(i), result(i), {"role": "assistant", "content": "vale " * 500}, user(f"sigue {i}")]
    return msgs


class T(unittest.TestCase):
    def test_skeleton_ignores_cache_control_and_tool_result_content(self):
        a = conv(3)
        b = conv(3)
        b[-1] = user("sigue 2", cache=True)
        b[2]["content"][0]["content"] = "[Old tool result content cleared]"
        self.assertEqual(c.skeleton_hash(a), c.skeleton_hash(b))
        b[-1] = user("otra cosa")
        self.assertNotEqual(c.skeleton_hash(a), c.skeleton_hash(b))

    def test_cut_prefers_real_prompt(self):
        m = conv(10)
        cut = c.choose_cut(m, 3000)
        self.assertEqual(m[cut]["role"], "user")
        self.assertFalse(c.has_tool_result(m[cut]))
        self.assertGreaterEqual(sum(c.est_msg(x) for x in m[cut:]), 3000)

    def test_cut_inside_long_tool_run(self):
        m = [user("haz algo largo")]
        for i in range(30):
            m += [asst_tool(i), result(i)]
        cut = c.choose_cut(m, 3000)
        self.assertIsNotNone(cut)
        self.assertTrue(c.has_tool_result(m[cut]))

    def test_apply_produces_valid_structure(self):
        m = [user("haz algo largo")]
        for i in range(30):
            m += [asst_tool(i), result(i)]
        cut = c.choose_cut(m, 3000)
        st = {"cut": cut, "hash": c.skeleton_hash(m[:cut + 1]), "summary": "RESUMEN"}
        m2 = m + [asst_tool(99), result(99)]          # la conversación sigue creciendo
        out = c.apply_state(m2, st)
        self.assertEqual(out[0]["role"], "user")
        types = [b["type"] for b in out[0]["content"]]
        self.assertNotIn("tool_result", types)        # sin tool_use previo no puede haber tool_result
        self.assertIn("RESUMEN", out[0]["content"][0]["text"])
        roles = [x["role"] for x in out]
        self.assertTrue(all(roles[i] != roles[i + 1] for i in range(len(roles) - 1)))
        ids = {b["id"] for x in out for b in c.blocks(x) if b.get("type") == "tool_use"}
        for x in out:
            for b in c.blocks(x):
                if b.get("type") == "tool_result":
                    self.assertIn(b["tool_use_id"], ids)
        self.assertLess(c.compactable_tokens(out), c.compactable_tokens(m2))

    def test_apply_keeps_first_message_reminders(self):
        m = conv(10)
        cut = c.choose_cut(m, 3000)
        st = {"cut": cut, "hash": c.skeleton_hash(m[:cut + 1]), "summary": "S"}
        out = c.apply_state(m, st)
        self.assertIn("CLAUDE.md", out[0]["content"][0]["text"])

    def test_apply_rejects_rewind(self):
        m = conv(10)
        cut = c.choose_cut(m, 3000)
        st = {"cut": cut, "hash": c.skeleton_hash(m[:cut + 1]), "summary": "S"}
        m2 = list(m)
        m2[3] = user("editado")
        self.assertIsNone(c.apply_state(m2, st))
        self.assertIsNone(c.apply_state(m[:cut], st))

    def test_transform_below_trigger_is_identity(self):
        b = {"messages": conv(1)}
        self.assertIs(c.transform(b, count_only=True), b)

    def test_render_and_files(self):
        m = conv(2)
        r = c.render(m)
        self.assertNotIn("CLAUDE.md", r)
        self.assertIn("[CLAUDE usa Read]", r)
        mod, read = c.touched_files(m)
        self.assertEqual(read, ["/f0.py", "/f1.py"])

    def test_chunks(self):
        parts = c.chunks("\n".join(["a" * 100] * 50), 1000)
        self.assertTrue(all(len(p) <= 1100 for p in parts))
        self.assertEqual(sum(p.count("a" * 100) for p in parts), 50)


if __name__ == "__main__":
    unittest.main()


def test_parse_usage_sse():
    sse = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":12,'
           b'"cache_creation_input_tokens":300,"cache_read_input_tokens":45000,"output_tokens":1}}}\n\n'
           b'event: content_block_delta\ndata: {"delta":{"text":"x \\"input_tokens\\": 999"}}\n\n'
           b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":420}}\n\n')
    u = c.parse_usage(sse)
    assert u == {"input_tokens": 12, "cache_read_input_tokens": 45000,
                 "cache_creation_input_tokens": 300, "output_tokens": 420}


def test_ratio_scales_pending():
    c.RATIO.clear()
    body = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                         {"role": "user", "content": "c"}], "tools": [{"name": "x" * 7000}]}
    est = c.est_body(body)
    c.record_usage({"input_tokens": 0, "cache_read_input_tokens": int(est * 3),
                    "cache_creation_input_tokens": 0, "output_tokens": 1}, False, "k", "conv1", est)
    assert abs(c.RATIO["conv1"] - 3.0) < 0.01
    c.record_usage({"input_tokens": 0, "cache_read_input_tokens": int(est * 50),
                    "cache_creation_input_tokens": 0, "output_tokens": 1}, False, "k", "conv1", est)
    assert c.RATIO["conv1"] == 5.0
    c.RATIO.clear()
