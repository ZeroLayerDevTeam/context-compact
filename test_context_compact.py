import unittest

import context_compact as c


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
        m2 = m + [asst_tool(99), result(99)]          # the conversation keeps growing
        out = c.apply_state(m2, st)
        self.assertEqual(out[0]["role"], "user")
        types = [b["type"] for b in out[0]["content"]]
        self.assertNotIn("tool_result", types)        # no tool_result without a preceding tool_use
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
        self.assertIn("[CLAUDE uses Read]", r)
        mod, read = c.touched_files(m)
        self.assertEqual(read, ["/f0.py", "/f1.py"])

    def test_chunks(self):
        parts = c.chunks("\n".join(["a" * 100] * 50), 1000)
        self.assertTrue(all(len(p) <= 1100 for p in parts))
        self.assertEqual(sum(p.count("a" * 100) for p in parts), 50)

    def test_parse_usage_sse(self):
        sse = (b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":12,'
               b'"cache_creation_input_tokens":300,"cache_read_input_tokens":45000,"output_tokens":1}}}\n\n'
               b'event: content_block_delta\ndata: {"delta":{"text":"x \\"input_tokens\\": 999"}}\n\n'
               b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":420}}\n\n')
        u = c.parse_usage(sse)
        assert u == {"input_tokens": 12, "cache_read_input_tokens": 45000,
                     "cache_creation_input_tokens": 300, "output_tokens": 420}

    def test_ratio_scales_pending(self):
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


    def _state(self, m, cut, **kw):
        return dict({"cut": cut, "hash": c.skeleton_hash(m[:cut + 1]), "summary": "S", "t": cut}, **kw)

    def test_trim_old_tool_results_and_thinking(self):
        m = conv(10)
        m[5]["content"].insert(0, {"type": "thinking", "thinking": "deep", "signature": "sig"})
        m[-3]["content"] = [{"type": "thinking", "thinking": "recent", "signature": "s2"},
                            {"type": "text", "text": "ok"}]
        m[6]["content"][0]["content"] = "z" * 10000
        cut = c.choose_cut(m, 3000)
        cut = min(cut, 4)
        tb, thb = c.trim_bounds(m, cut)
        self.assertLessEqual(thb, tb)
        out = c.apply_state(m, self._state(m, cut, trim_before=tb, think_before=thb))
        flat = [b for x in out for b in (x["content"] if isinstance(x["content"], list) else [])]
        trimmed = [b for b in flat if b.get("type") == "tool_result" and "chars trimmed" in str(b["content"])]
        self.assertTrue(trimmed)
        self.assertNotIn("deep", str(out))           # finished turn: thinking removed
        self.assertIn("recent", str(out))            # newest messages untouched
        self.assertEqual(out, c.apply_state(m, self._state(m, cut, trim_before=tb, think_before=thb)))
        self.assertIn("z" * 10000, str(m))           # original history not mutated

    def test_trim_keeps_assistant_with_only_thinking(self):
        m = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"}]}]
        self.assertEqual(c.trim_old(m, 0, 5, 5)[0]["content"], m[0]["content"])

    def test_best_state_picks_largest_matching_cut(self):
        m = conv(10)
        old, new = self._state(m, 8), self._state(m, 20)
        self.assertIs(c.best_state(m, [old, new]), new)
        fork = m[:15] + [user("otra rama")]
        self.assertIs(c.best_state(fork, [old, new]), old)
        self.assertIsNone(c.best_state(m[:5], [old, new]))

    def test_facts_and_coverage(self):
        frag = ('[CLAUDE uses Read] {"file_path": "G:\\\\Proyectos\\\\x\\\\a.py"}\n'
                "[result ERROR] Exit code 2 permission denied\n"
                "see https://gitlab.com/zerolayer/context-compact/-/pipelines/1. and /home/servidor/x.log")
        fs = c.facts(frag)
        self.assertIn("G:\\Proyectos\\x\\a.py", fs)
        self.assertIn("https://gitlab.com/zerolayer/context-compact/-/pipelines/1", fs)
        self.assertIn("/home/servidor/x.log", fs)
        self.assertIn("Exit code 2 permission denied", fs)
        miss, cov = c.missing_facts(frag, "touched G:\\Proyectos\\x\\a.py")
        self.assertEqual(len(miss), 3)
        self.assertAlmostEqual(cov, 0.25)

    def test_early_summary_waits_for_trigger(self):
        m = conv(10)
        cut = c.choose_cut(m, 3000)
        st = {"cut": cut, "hash": c.skeleton_hash(m[:cut + 1]), "summary": "S", "base": 0}
        total = c.compactable_tokens(m)
        old = c.TRIGGER
        try:
            c.TRIGGER = total + 1          # history still below the trigger: not armed yet
            self.assertFalse(c.armed(m, st))
            c.TRIGGER = total - 1
            self.assertTrue(c.armed(m, st))
            self.assertTrue(c.armed(m, {"cut": cut}))   # states without base: always armed
        finally:
            c.TRIGGER = old

    def test_header_mentions_conversation_with_mcp(self):
        m = conv(10)
        cut = c.choose_cut(m, 3000)
        st = {"cut": cut, "hash": c.skeleton_hash(m[:cut + 1]), "summary": "S"}
        txt = c.apply_state(m, st, "abc123def456")[0]["content"][1]["text"]
        self.assertIn('conversation="abc123def456"', txt)
        self.assertNotIn("history_search", c.apply_state(m, st)[0]["content"][1]["text"])

    def test_search_entries(self):
        m = [user("arranca el servicio"),
             {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                                "input": {"command": "systemctl restart foo"}}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                                           "content": "Job for foo.service failed: port 8080 in use"}]},
             user("vale, usa el puerto 9090")]
        e = c.archive_entries(m)
        self.assertEqual([x[0] for x in e], [0, 1, 2, 3])
        out = c.search_entries(e, "port 8080")
        self.assertIn("message 2", out)
        self.assertIn("[result ERROR]", out)
        self.assertIn("No match", c.search_entries(e, "kubernetes"))

    def test_mcp_protocol(self):
        init = c.mcp_handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                             "params": {"protocolVersion": "2025-06-18"}})
        self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
        self.assertIsNone(c.mcp_handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        tools = c.mcp_handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        self.assertEqual(tools[0]["name"], "history_search")
        r = c.mcp_handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                          "params": {"name": "history_search", "arguments": {"conversation": "zz", "query": "x"}}})
        self.assertIn("No summarised history", r["result"]["content"][0]["text"])
        self.assertIn("error", c.mcp_handle({"jsonrpc": "2.0", "id": 4, "method": "nope"}))

    def test_parse_extract(self):
        out = c.parse_extract('{"paths": ["/etc/x.conf"], "commands": "make", "pending": [1, {"a": 1}], "x": ["y"]}')
        self.assertEqual(out, ["paths: /etc/x.conf", "commands: make", "pending: 1"])
        self.assertEqual(c.parse_extract("not json"), [])


if __name__ == "__main__":
    unittest.main()
