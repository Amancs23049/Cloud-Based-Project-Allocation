"""
Project 39: AI-Based Cloud Resource Allocation System
S.B. Jain Institute of Technology, Management & Research, Nagpur
Department of Computer Science and Engineering

AI Technique: Heuristic Search (A* + Greedy Best-First Search)
Representation: Resource Graph
Tools: Python, Google OR-Tools (simulated), Flask
"""

from flask import Flask, request, jsonify, send_from_directory
import heapq
import time
import random
import math
import os

app = Flask(__name__, static_folder='static')

# ─────────────────────────────────────────────
#  DATA MODEL
# ─────────────────────────────────────────────

class VM:
    def __init__(self, vm_id, cpu_required, memory_required, priority, task_name):
        self.vm_id        = vm_id
        self.cpu_required = cpu_required          # cores
        self.memory_required = memory_required    # GB
        self.priority     = priority              # 1=high, 2=medium, 3=low
        self.task_name    = task_name

class Server:
    def __init__(self, server_id, cpu_total, memory_total, cost_per_unit):
        self.server_id      = server_id
        self.cpu_total      = cpu_total
        self.memory_total   = memory_total
        self.cost_per_unit  = cost_per_unit       # $ per allocated unit
        self.cpu_used       = 0
        self.memory_used    = 0
        self.allocated_vms  = []

    @property
    def cpu_free(self):
        return self.cpu_total - self.cpu_used

    @property
    def memory_free(self):
        return self.memory_total - self.memory_used

    @property
    def utilization(self):
        cpu_util = self.cpu_used / self.cpu_total if self.cpu_total > 0 else 0
        mem_util = self.memory_used / self.memory_total if self.memory_total > 0 else 0
        return (cpu_util + mem_util) / 2

    def can_fit(self, vm):
        return self.cpu_free >= vm.cpu_required and self.memory_free >= vm.memory_required

    def allocate(self, vm):
        self.cpu_used    += vm.cpu_required
        self.memory_used += vm.memory_required
        self.allocated_vms.append(vm.vm_id)

    def clone(self):
        s = Server(self.server_id, self.cpu_total, self.memory_total, self.cost_per_unit)
        s.cpu_used    = self.cpu_used
        s.memory_used = self.memory_used
        s.allocated_vms = list(self.allocated_vms)
        return s

    def to_dict(self):
        return {
            "server_id":      self.server_id,
            "cpu_total":      self.cpu_total,
            "memory_total":   self.memory_total,
            "cpu_used":       self.cpu_used,
            "memory_used":    self.memory_used,
            "cpu_free":       self.cpu_free,
            "memory_free":    self.memory_free,
            "utilization_pct": round(self.utilization * 100, 1),
            "cost_per_unit":  self.cost_per_unit,
            "allocated_vms":  self.allocated_vms
        }


# ─────────────────────────────────────────────
#  HEURISTIC FUNCTIONS
# ─────────────────────────────────────────────

def heuristic_cost(vm, server):
    """
    Heuristic: lower score = better server for this VM.
    Combines cost, remaining capacity after allocation, and utilization balance.
    """
    if not server.can_fit(vm):
        return float('inf')

    # Cost component
    allocation_cost = (vm.cpu_required + vm.memory_required) * server.cost_per_unit

    # Waste penalty: prefer servers that will be well-utilized after allocation
    cpu_waste    = (server.cpu_free - vm.cpu_required) / server.cpu_total
    memory_waste = (server.memory_free - vm.memory_required) / server.memory_total
    waste_penalty = (cpu_waste + memory_waste) * 10

    # Priority bonus: high-priority VMs get better (costlier/faster) servers
    priority_bonus = vm.priority * 2

    return allocation_cost + waste_penalty + priority_bonus


# ─────────────────────────────────────────────
#  GREEDY BEST-FIRST SEARCH
# ─────────────────────────────────────────────

def greedy_best_first_search(vms, servers):
    """
    Greedy Best-First Search: always picks the server with lowest heuristic
    for each VM. Fast but not guaranteed optimal.
    """
    start_time   = time.time()
    nodes_explored = 0
    allocation_log = []
    total_cost   = 0.0

    # Sort VMs by priority (high priority first)
    sorted_vms = sorted(vms, key=lambda v: v.priority)

    server_states = [s.clone() for s in servers]

    for vm in sorted_vms:
        # Priority queue: (heuristic_score, server_index)
        pq = []
        for i, s in enumerate(server_states):
            score = heuristic_cost(vm, s)
            heapq.heappush(pq, (score, i))
            nodes_explored += 1

        if pq:
            best_score, best_idx = heapq.heappop(pq)
            if best_score < float('inf'):
                chosen_server = server_states[best_idx]
                cost = (vm.cpu_required + vm.memory_required) * chosen_server.cost_per_unit
                total_cost += cost
                chosen_server.allocate(vm)
                allocation_log.append({
                    "vm_id":     vm.vm_id,
                    "task":      vm.task_name,
                    "server_id": chosen_server.server_id,
                    "cpu":       vm.cpu_required,
                    "memory":    vm.memory_required,
                    "cost":      round(cost, 2),
                    "score":     round(best_score, 2),
                    "status":    "Allocated"
                })
            else:
                allocation_log.append({
                    "vm_id":     vm.vm_id,
                    "task":      vm.task_name,
                    "server_id": None,
                    "cpu":       vm.cpu_required,
                    "memory":    vm.memory_required,
                    "cost":      0,
                    "score":     float('inf'),
                    "status":    "Failed – No capacity"
                })

    exec_time = round((time.time() - start_time) * 1000, 3)

    return {
        "algorithm":      "Greedy Best-First Search",
        "allocation_log": allocation_log,
        "total_cost":     round(total_cost, 2),
        "nodes_explored": nodes_explored,
        "exec_time_ms":   exec_time,
        "server_states":  [s.to_dict() for s in server_states]
    }


# ─────────────────────────────────────────────
#  A* SEARCH
# ─────────────────────────────────────────────

def astar_search(vms, servers):
    """
    A* Search: f(n) = g(n) + h(n)
    g(n) = actual cost so far
    h(n) = heuristic estimate of remaining cost
    Optimal but explores more nodes than greedy.
    """
    start_time     = time.time()
    nodes_explored = 0
    allocation_log = []
    total_cost     = 0.0

    sorted_vms    = sorted(vms, key=lambda v: v.priority)
    server_states = [s.clone() for s in servers]

    for vm in sorted_vms:
        # State: (f_score, g_score, server_index)
        open_list = []
        for i, s in enumerate(server_states):
            if s.can_fit(vm):
                g = (vm.cpu_required + vm.memory_required) * s.cost_per_unit
                h = heuristic_cost(vm, s)
                f = g + h * 0.5   # weighted A*
                heapq.heappush(open_list, (f, g, i))
            nodes_explored += 1

        if open_list:
            f_best, g_best, best_idx = heapq.heappop(open_list)
            chosen = server_states[best_idx]
            cost   = (vm.cpu_required + vm.memory_required) * chosen.cost_per_unit
            total_cost += cost
            chosen.allocate(vm)
            allocation_log.append({
                "vm_id":     vm.vm_id,
                "task":      vm.task_name,
                "server_id": chosen.server_id,
                "cpu":       vm.cpu_required,
                "memory":    vm.memory_required,
                "cost":      round(cost, 2),
                "score":     round(f_best, 2),
                "status":    "Allocated"
            })
        else:
            allocation_log.append({
                "vm_id":     vm.vm_id,
                "task":      vm.task_name,
                "server_id": None,
                "cpu":       vm.cpu_required,
                "memory":    vm.memory_required,
                "cost":      0,
                "score":     float('inf'),
                "status":    "Failed – No capacity"
            })

    exec_time = round((time.time() - start_time) * 1000, 3)

    return {
        "algorithm":      "A* Search",
        "allocation_log": allocation_log,
        "total_cost":     round(total_cost, 2),
        "nodes_explored": nodes_explored,
        "exec_time_ms":   exec_time,
        "server_states":  [s.to_dict() for s in server_states]
    }


# ─────────────────────────────────────────────
#  DEFAULT DATASET
# ─────────────────────────────────────────────

DEFAULT_SERVERS = [
    Server("S1", cpu_total=32, memory_total=128, cost_per_unit=0.05),
    Server("S2", cpu_total=16, memory_total=64,  cost_per_unit=0.03),
    Server("S3", cpu_total=64, memory_total=256, cost_per_unit=0.08),
    Server("S4", cpu_total=8,  memory_total=32,  cost_per_unit=0.02),
]

DEFAULT_VMS = [
    VM("VM1",  cpu_required=4,  memory_required=16, priority=1, task_name="Web Server"),
    VM("VM2",  cpu_required=8,  memory_required=32, priority=1, task_name="Database"),
    VM("VM3",  cpu_required=2,  memory_required=8,  priority=2, task_name="Cache Layer"),
    VM("VM4",  cpu_required=16, memory_required=64, priority=1, task_name="ML Training"),
    VM("VM5",  cpu_required=4,  memory_required=16, priority=3, task_name="Batch Job"),
    VM("VM6",  cpu_required=2,  memory_required=4,  priority=2, task_name="Monitoring"),
    VM("VM7",  cpu_required=8,  memory_required=32, priority=2, task_name="API Gateway"),
    VM("VM8",  cpu_required=1,  memory_required=2,  priority=3, task_name="Log Collector"),
]


# ─────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/default-data', methods=['GET'])
def get_default_data():
    return jsonify({
        "servers": [s.to_dict() for s in DEFAULT_SERVERS],
        "vms": [{"vm_id": v.vm_id, "task_name": v.task_name,
                 "cpu_required": v.cpu_required, "memory_required": v.memory_required,
                 "priority": v.priority} for v in DEFAULT_VMS]
    })

@app.route('/api/allocate', methods=['POST'])
def allocate():
    data = request.get_json()

    # Parse servers
    servers = []
    for s in data.get('servers', []):
        servers.append(Server(
            s['server_id'], s['cpu_total'], s['memory_total'], s['cost_per_unit']
        ))

    # Parse VMs
    vms = []
    for v in data.get('vms', []):
        vms.append(VM(
            v['vm_id'], v['cpu_required'], v['memory_required'],
            v['priority'], v['task_name']
        ))

    algorithm = data.get('algorithm', 'astar')

    if algorithm == 'greedy':
        result = greedy_best_first_search(vms, servers)
    else:
        result = astar_search(vms, servers)

    return jsonify(result)

@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.get_json()

    servers_data = data.get('servers', [s.__dict__ for s in DEFAULT_SERVERS])
    vms_data     = data.get('vms',     [])

    servers1 = [Server(s['server_id'], s['cpu_total'], s['memory_total'], s['cost_per_unit']) for s in servers_data]
    servers2 = [Server(s['server_id'], s['cpu_total'], s['memory_total'], s['cost_per_unit']) for s in servers_data]

    vms = [VM(v['vm_id'], v['cpu_required'], v['memory_required'], v['priority'], v['task_name']) for v in vms_data] if vms_data else DEFAULT_VMS

    greedy_result = greedy_best_first_search(vms, servers1)
    astar_result  = astar_search(vms, servers2)

    return jsonify({
        "greedy": greedy_result,
        "astar":  astar_result
    })

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("=" * 55)
    print("  Project 39: AI Cloud Resource Allocation System")
    print("  S.B. Jain Institute of Technology, Nagpur")
    print("  Running at http://127.0.0.1:5000")
    print("=" * 55)
    port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)