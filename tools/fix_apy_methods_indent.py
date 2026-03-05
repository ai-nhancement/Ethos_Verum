"""Move write_apy_context/get_apy_context/prune_apy_context inside ValueStore class."""

with open("core/value_store.py", encoding="utf-8") as f:
    src = f.read()

# The misplaced block (at module level, 4-space indent)
old_misplaced = (
    "\n"
    "    # ------------------------------------------------------------------\n"
    "    # APY Context (cross-passage pressure window)\n"
    "    # ------------------------------------------------------------------\n"
    "\n"
    "    def write_apy_context(\n"
    "        self,\n"
    "        session_id: str,\n"
    "        record_id: str,\n"
    "        ts: float,\n"
    "        passage_idx: int,\n"
    "        markers: str,\n"
    "        window_n: int = 5,\n"
    "    ) -> None:\n"
    '        """Write a pressure context entry and prune to the N most recent."""\n'
    "        import uuid as _uuid\n"
    "        try:\n"
    "            conn = self._conn()\n"
    "            conn.execute(\n"
    '                """INSERT OR REPLACE INTO apy_context\n'
    "                   (id, session_id, record_id, ts, passage_idx, markers)\n"
    '                   VALUES (?,?,?,?,?,?)""",\n'
    "                (str(_uuid.uuid4()), session_id, record_id, float(ts),\n"
    "                 int(passage_idx), str(markers)),\n"
    "            )\n"
    "            # Prune: keep only the N most recent by passage_idx\n"
    "            conn.execute(\n"
    '                """DELETE FROM apy_context\n'
    "                   WHERE session_id=?\n"
    "                   AND id NOT IN (\n"
    "                       SELECT id FROM apy_context\n"
    "                       WHERE session_id=?\n"
    "                       ORDER BY passage_idx DESC, ts DESC\n"
    "                       LIMIT ?\n"
    "                   )\"\"\",\n"
    "                (session_id, session_id, int(window_n)),\n"
    "            )\n"
    "            conn.commit()\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "    def get_apy_context(\n"
    "        self,\n"
    "        session_id: str,\n"
    "        since_passage_idx: int = 0,\n"
    "        since_ts: float = 0.0,\n"
    "    ) -> List[Dict]:\n"
    '        """Return recent pressure context entries for a session."""\n'
    "        try:\n"
    "            conn = self._conn()\n"
    "            rows = conn.execute(\n"
    '                """SELECT record_id, ts, passage_idx, markers\n'
    "                   FROM apy_context\n"
    "                   WHERE session_id=?\n"
    "                     AND (passage_idx >= ? OR ts >= ?)\n"
    '                   ORDER BY passage_idx DESC""",\n'
    "                (session_id, int(since_passage_idx), float(since_ts)),\n"
    "            ).fetchall()\n"
    "            return [dict(r) for r in rows]\n"
    "        except Exception:\n"
    "            return []\n"
    "\n"
    "    def prune_apy_context(self, session_id: str, keep_n: int = 5) -> None:\n"
    '        """Keep only the N most recent pressure context entries."""\n'
    "        try:\n"
    "            conn = self._conn()\n"
    "            conn.execute(\n"
    '                """DELETE FROM apy_context\n'
    "                   WHERE session_id=?\n"
    "                   AND id NOT IN (\n"
    "                       SELECT id FROM apy_context\n"
    "                       WHERE session_id=?\n"
    "                       ORDER BY passage_idx DESC, ts DESC\n"
    "                       LIMIT ?\n"
    "                   )\"\"\",\n"
    "                (session_id, session_id, int(keep_n)),\n"
    "            )\n"
    "            conn.commit()\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
)

assert src.count(old_misplaced) == 1, f"misplaced block not found uniquely (count={src.count(old_misplaced)})"
src = src.replace(old_misplaced, "\n")  # remove the misplaced block

# Now insert the methods INSIDE the class, after get_stats()
# Anchor: the blank line that ends get_stats + the module-level comment line
old_class_end = (
    "        except Exception:\n"
    "            return {\"total_observations\": 0, \"top_values\": [], \"figure_count\": 0}\n"
    "\n"
    "\n"
    "# ------------------------------------------------------------------\n"
    "# Helpers\n"
)
new_class_end = (
    "        except Exception:\n"
    "            return {\"total_observations\": 0, \"top_values\": [], \"figure_count\": 0}\n"
    "\n"
    "    # ------------------------------------------------------------------\n"
    "    # APY Context (cross-passage pressure window)\n"
    "    # ------------------------------------------------------------------\n"
    "\n"
    "    def write_apy_context(\n"
    "        self,\n"
    "        session_id: str,\n"
    "        record_id: str,\n"
    "        ts: float,\n"
    "        passage_idx: int,\n"
    "        markers: str,\n"
    "        window_n: int = 5,\n"
    "    ) -> None:\n"
    '        """Write a pressure context entry and prune to the N most recent."""\n'
    "        import uuid as _uuid\n"
    "        try:\n"
    "            conn = self._conn()\n"
    "            conn.execute(\n"
    '                """INSERT OR REPLACE INTO apy_context\n'
    "                   (id, session_id, record_id, ts, passage_idx, markers)\n"
    '                   VALUES (?,?,?,?,?,?)""",\n'
    "                (str(_uuid.uuid4()), session_id, record_id, float(ts),\n"
    "                 int(passage_idx), str(markers)),\n"
    "            )\n"
    "            # Prune: keep only the N most recent by passage_idx\n"
    "            conn.execute(\n"
    '                """DELETE FROM apy_context\n'
    "                   WHERE session_id=?\n"
    "                   AND id NOT IN (\n"
    "                       SELECT id FROM apy_context\n"
    "                       WHERE session_id=?\n"
    "                       ORDER BY passage_idx DESC, ts DESC\n"
    "                       LIMIT ?\n"
    "                   )\"\"\",\n"
    "                (session_id, session_id, int(window_n)),\n"
    "            )\n"
    "            conn.commit()\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "    def get_apy_context(\n"
    "        self,\n"
    "        session_id: str,\n"
    "        since_passage_idx: int = 0,\n"
    "        since_ts: float = 0.0,\n"
    "    ) -> List[Dict]:\n"
    '        """Return recent pressure context entries for a session."""\n'
    "        try:\n"
    "            conn = self._conn()\n"
    "            rows = conn.execute(\n"
    '                """SELECT record_id, ts, passage_idx, markers\n'
    "                   FROM apy_context\n"
    "                   WHERE session_id=?\n"
    "                     AND (passage_idx >= ? OR ts >= ?)\n"
    '                   ORDER BY passage_idx DESC""",\n'
    "                (session_id, int(since_passage_idx), float(since_ts)),\n"
    "            ).fetchall()\n"
    "            return [dict(r) for r in rows]\n"
    "        except Exception:\n"
    "            return []\n"
    "\n"
    "    def prune_apy_context(self, session_id: str, keep_n: int = 5) -> None:\n"
    '        """Keep only the N most recent pressure context entries."""\n'
    "        try:\n"
    "            conn = self._conn()\n"
    "            conn.execute(\n"
    '                """DELETE FROM apy_context\n'
    "                   WHERE session_id=?\n"
    "                   AND id NOT IN (\n"
    "                       SELECT id FROM apy_context\n"
    "                       WHERE session_id=?\n"
    "                       ORDER BY passage_idx DESC, ts DESC\n"
    "                       LIMIT ?\n"
    "                   )\"\"\",\n"
    "                (session_id, session_id, int(keep_n)),\n"
    "            )\n"
    "            conn.commit()\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "\n"
    "# ------------------------------------------------------------------\n"
    "# Helpers\n"
)

assert src.count(old_class_end) == 1, f"class_end not found (count={src.count(old_class_end)})"
src = src.replace(old_class_end, new_class_end)

with open("core/value_store.py", "w", encoding="utf-8") as f:
    f.write(src)

import py_compile
py_compile.compile("core/value_store.py", doraise=True)
print("value_store.py fixed and syntax OK")
