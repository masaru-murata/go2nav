"""Topological memory without a map (design chapter 8): a list of junction nodes, edges and notes."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Node:
    id: str
    turn: int
    t: float
    images: list[str] = field(default_factory=list)   # relative paths of saved images
    where_am_i: str = ""
    free_space: dict = field(default_factory=dict)
    stopped_because: str = ""
    chosen: str = ""


@dataclass
class Edge:
    frm: str
    to: str
    turn_deg: float
    dist_m: float


@dataclass
class Note:
    turn: int
    kind: str
    note: str


class Memory:
    def __init__(self):
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.notes: list[Note] = []
        self.current: Optional[str] = None
        self.came_from: Optional[str] = None
        self.pending_turn_deg = 0.0        # accumulated turning since the current node was created
        self.visits: dict[str, int] = {}

    # -- nodes ----------------------------------------------------------------------
    def add_node(self, turn: int, t: float, images: list[str], free_space: dict, stopped_because: str, dist_m: float,
                 where_am_i: str = "") -> Node:
        nid = f"n{len(self.nodes)}"
        n = Node(id=nid, turn=turn, t=t, images=list(images), where_am_i=where_am_i, free_space=dict(free_space),
                 stopped_because=stopped_because)
        if self.current is not None:
            self.edges.append(Edge(frm=self.current, to=nid, turn_deg=round(self.pending_turn_deg, 1), dist_m=round(dist_m, 2)))
        self.came_from, self.current = self.current, nid
        self.pending_turn_deg = 0.0
        self.nodes.append(n)
        self.visits[nid] = 1
        return n

    def note_turn(self, deg: float) -> None:
        self.pending_turn_deg += deg

    def set_where(self, where: str) -> None:
        if self.current:
            self.node(self.current).where_am_i = where

    def set_chosen(self, chosen: str) -> None:
        if self.current:
            self.node(self.current).chosen = chosen

    def node(self, nid: str) -> Node:
        return next(n for n in self.nodes if n.id == nid)

    def add_note(self, turn: int, kind: str, note: str) -> None:
        self.notes.append(Note(turn=turn, kind=kind, note=note))

    # -- backtracking ---------------------------------------------------------------
    def backtrack_plan(self, steps: int) -> tuple[list[tuple[str, dict]], Optional[str]]:
        """Primitives to walk `steps` edges back. Each leg: turn(around) -> go the recorded distance -> undo the recorded turn.
        Returns (plan, target_node_id)."""
        plan: list[tuple[str, dict]] = []
        cur = self.current
        target = cur
        for _ in range(max(1, int(steps))):
            e = next((e for e in reversed(self.edges) if e.to == cur), None)
            if e is None:
                break
            plan.append(("turn", {"bearing": "around"}))
            plan.append(("go_forward_until", {"stop_on": ["blocked", "distance"], "max_m": max(0.5, e.dist_m)}))
            if abs(e.turn_deg) >= 10.0:
                plan.append(("turn", {"bearing": round(e.turn_deg, 1)}))   # same sign restores the arrival heading (see design 8)
            cur = e.frm
            target = cur
        if target == self.current:
            return [], None
        return plan, target

    def arrive_back(self, nid: str) -> None:
        self.came_from, self.current = self.current, nid
        self.pending_turn_deg = 0.0
        self.visits[nid] = self.visits.get(nid, 0) + 1

    # -- summaries ------------------------------------------------------------------
    def summary(self) -> dict:
        return {"current_node": self.current, "came_from": self.came_from, "nodes_total": len(self.nodes),
                "visits_here": self.visits.get(self.current, 0) if self.current else 0}

    def notebook_tail(self, n: int = 6) -> list[str]:
        return [f"[{x.kind}] {x.note}" for x in self.notes[-n:]]

    def route(self) -> list[dict]:
        """Path from the start node to the current node as a list of legs (for notify / report)."""
        if not self.nodes:
            return []
        parent = {e.to: e for e in self.edges}
        legs = []
        cur = self.current
        while cur is not None and cur in parent:
            e = parent[cur]
            n = self.node(e.to)
            legs.append({"from": e.frm, "to": e.to, "turn_deg": e.turn_deg, "dist_m": e.dist_m,
                         "arrived_because": n.stopped_because, "where": n.where_am_i, "images": list(n.images)})
            cur = e.frm
        legs.reverse()
        return legs

    def route_text(self) -> str:
        start = self.nodes[0].where_am_i if self.nodes else "出発点"
        parts = [f"出発点({start})"]
        for leg in self.route():
            d = leg["turn_deg"]
            turn = "直進" if abs(d) < 10 else (f"左へ{abs(d):.0f}°" if d > 0 else f"右へ{abs(d):.0f}°")
            parts.append(f"{turn}→{leg['dist_m']:.1f}m（{leg['arrived_because']}{'、' + leg['where'] if leg['where'] else ''}）")
        return " → ".join(parts)

    def to_dict(self) -> dict:
        return {"nodes": [asdict(n) for n in self.nodes], "edges": [asdict(e) for e in self.edges],
                "notes": [asdict(n) for n in self.notes], "current": self.current, "came_from": self.came_from,
                "visits": self.visits}
