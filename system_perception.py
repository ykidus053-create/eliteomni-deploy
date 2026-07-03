import subprocess, os, psutil, logging
log = logging.getLogger(__name__)

def get_os_state() -> str:
    """Upgraded: AI can see active ports, memory, and CPU to debug infra issues."""
    try:
        state = ["[OS STATE]"]
        
        # Memory and CPU
        vm = psutil.virtual_memory()
        state.append(f"Memory: {vm.percent}% used ({vm.available // (1024**2)}MB available)")
        state.append(f"CPU: {psutil.cpu_percent(interval=0.1)}%")
        
        # Active listening ports (crucial for debugging 'Address already in use')
        connections = psutil.net_connections(kind='inet')
        listening = [c.laddr.port for c in connections if c.status == 'LISTEN']
        if listening:
            state.append(f"Active Ports: {sorted(list(set(listening)))}")
            
        # Top 3 CPU consuming processes
        procs = []
        for p in psutil.process_iter(['name', 'cpu_percent']):
            try:
                if p.info['cpu_percent'] > 0:
                    procs.append((p.info['cpu_percent'], p.info['name']))
            except: pass
        procs.sort(reverse=True)
        if procs:
            state.append("Top Processes: " + ", ".join(f"{n} ({c}%)" for c, n in procs[:3]))
            
        state.append("[END OS STATE]")
        return "\n".join(state)
    except Exception as e:
        return ""
