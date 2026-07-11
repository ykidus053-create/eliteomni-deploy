import importlib


CASES = [
    ("proactive_daemon", "start_proactive_daemon", (), "ELITE_ENABLE_PROACTIVE_DAEMON"),
    ("agi_emulation_layer", "start_agi_emulation", (lambda *a, **k: "",), "ELITE_ENABLE_AGI_EMULATION"),
    ("refactor_daemon", "start_refactor_daemon", (lambda *a, **k: "",), "ELITE_ENABLE_REFACTOR_DAEMON"),
    ("self_healing", "start_self_healing_daemon", (lambda *a, **k: "",), "ELITE_ENABLE_SELF_HEALING"),
]


def test_background_daemons_are_disabled_by_default(monkeypatch):
    for module_name, function_name, args, env_name in CASES:
        monkeypatch.delenv(env_name, raising=False)
        module = importlib.import_module(module_name)
        result = getattr(module, function_name)(*args)
        assert result is None
