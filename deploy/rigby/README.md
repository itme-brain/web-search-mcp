# Rigby deployment overrides

These files record the production host configuration. The Compose override
belongs at the repository root on Rigby; authentication files stay in
`/etc/web-search-mcp` and are not committed.

Valkey Search 1.0.3 can deadlock during forked background snapshots while
servicing searches (upstream valkey-search#1292). The override disables
automatic background saves and AOF. The systemd timer runs a synchronous
`SAVE` every five minutes instead. This preserves snapshots without forking;
each save briefly blocks cache operations (under one second when deployed).

Install the service and timer in `/etc/systemd/system`, run
`systemctl daemon-reload`, and enable `web-search-valkey-snapshot.timer`.
Keep the timer enabled while background saves are disabled.
