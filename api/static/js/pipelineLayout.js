// Auto-layout for the pipeline canvas. There is no x/y field anywhere in
// the backend's PipelineDefinition schema — positions are NOT part of the
// saved data model, so this computes a fresh layered layout every time
// from graph structure alone (same walk the backend's reachability
// validator does). Nodes are draggable in the session for readability,
// but dragging never persists — that would be storing fake state the
// server has nowhere to keep.

export const COL_W = 260;
export const ROW_H = 132;
export const NODE_W = 210;
export const NODE_H = 88;
export const PAD_X = 40;
export const PAD_Y = 40;

// Walks forward from entry_step exactly like custom_validator.py's
// _check_reachability, assigning each reached step a column = its
// distance from entry. Steps that share a column get stacked in rows.
// A decision outcome that points at a step whose column is <= the
// decision's own column is a loop-back (or self-loop) — flagged for the
// dashed/curved edge treatment, same "is_backward" concept the real
// validator uses to require is_loop_back.
export function computeLayout(def) {
  const stepById = new Map(def.steps.map((s) => [s.id, s]));
  const order = def.steps.map((s) => s.id);
  const indexOf = new Map(order.map((id, i) => [id, i]));

  const nextOf = (step) => {
    if (step.type === 'decision') return null; // handled via outcomes, not a single "next"
    const override = def.edge_overrides[step.id];
    if (override) return override === '__end__' ? null : override;
    const idx = indexOf.get(step.id);
    return idx + 1 < order.length ? order[idx + 1] : null;
  };

  const column = new Map();
  const visited = new Set();
  const queue = [[def.entry_step, 0]];
  while (queue.length) {
    const [id, col] = queue.shift();
    if (!stepById.has(id)) continue;
    if (visited.has(id) && column.get(id) <= col) continue;
    visited.add(id);
    column.set(id, Math.max(column.get(id) ?? 0, col));

    const step = stepById.get(id);
    if (step.type === 'decision') {
      for (const outcome of step.outcomes || []) {
        const target = outcome.next_step;
        if (target && target !== '__end__' && stepById.has(target)) {
          const targetCol = column.get(target);
          const isBack = targetCol !== undefined && targetCol <= col;
          if (!isBack) queue.push([target, col + 1]);
        }
      }
    } else {
      const nxt = nextOf(step);
      if (nxt && stepById.has(nxt)) queue.push([nxt, col + 1]);
    }
  }

  // Anything not reached (typo'd edge, freshly-added orphan step) still
  // needs a position so the user can see and fix it — same signal as the
  // validator's "unreachable" error, made visually obvious instead of
  // only appearing as text.
  const unplacedCol = Math.max(0, ...[...column.values()]) + 2;
  let unplacedRow = 0;
  for (const step of def.steps) {
    if (!column.has(step.id)) {
      column.set(step.id, unplacedCol);
      step._unplacedRow = unplacedRow++;
    }
  }

  // Stack multiple steps sharing a column into rows.
  const colBuckets = new Map();
  for (const step of def.steps) {
    const col = column.get(step.id);
    if (!colBuckets.has(col)) colBuckets.set(col, []);
    colBuckets.get(col).push(step);
  }

  const positions = new Map();
  for (const [col, steps] of colBuckets) {
    steps.forEach((step, row) => {
      positions.set(step.id, {
        x: PAD_X + col * COL_W,
        y: PAD_Y + row * ROW_H,
        col, row,
        unreached: !visited.has(step.id),
      });
    });
  }

  const maxCol = Math.max(0, ...[...positions.values()].map((p) => p.col));
  const maxRow = Math.max(0, ...[...positions.values()].map((p) => p.row));
  const width = PAD_X * 2 + (maxCol + 1) * COL_W;
  const height = PAD_Y * 2 + (maxRow + 1) * ROW_H;

  return { positions, width, height, visited };
}

// Builds the list of edges to draw: forward step-order/override edges for
// existing+freeform steps, one edge per outcome for decision steps.
// Each edge carries whether it's a loop-back (dashed/curved) and, for
// decision edges, the outcome label to render.
export function computeEdges(def, positions) {
  const stepById = new Map(def.steps.map((s) => [s.id, s]));
  const order = def.steps.map((s) => s.id);
  const indexOf = new Map(order.map((id, i) => [id, i]));
  const edges = [];

  for (const step of def.steps) {
    if (step.type === 'decision') {
      for (const outcome of step.outcomes || []) {
        const target = outcome.next_step;
        if (target === '__end__') {
          edges.push({ from: step.id, to: null, label: outcome.value, isEnd: true, isLoopBack: false });
          continue;
        }
        if (!stepById.has(target)) continue; // dangling — validator will flag it, canvas just skips drawing
        const fromPos = positions.get(step.id);
        const toPos = positions.get(target);
        const isLoopBack = !!(fromPos && toPos && toPos.col <= fromPos.col);
        edges.push({ from: step.id, to: target, label: outcome.value, isEnd: false, isLoopBack });
      }
    } else {
      const override = def.edge_overrides[step.id];
      let target;
      if (override) {
        target = override === '__end__' ? null : override;
      } else {
        const idx = indexOf.get(step.id);
        target = idx + 1 < order.length ? order[idx + 1] : null;
      }
      if (target === null) {
        edges.push({ from: step.id, to: null, label: null, isEnd: true, isLoopBack: false });
      } else if (stepById.has(target)) {
        const fromPos = positions.get(step.id);
        const toPos = positions.get(target);
        const isLoopBack = !!(fromPos && toPos && toPos.col <= fromPos.col);
        edges.push({ from: step.id, to: target, label: null, isEnd: false, isLoopBack });
      }
    }
  }
  return edges;
}

// SVG path for one edge, anchored to node edges rather than centers.
// Loop-back edges curve below/above the row so they read as "going
// backward" rather than overlapping the forward flow.
export function edgePath(fromPos, toPos, isLoopBack, edgeIndex = 0) {
  const x1 = fromPos.x + NODE_W;
  const y1 = fromPos.y + NODE_H / 2;
  const x2 = toPos.x;
  const y2 = toPos.y + NODE_H / 2;

  if (!isLoopBack) {
    const midX = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`;
  }

  // Loop-back: drop below both nodes and curve back, offset per-edge so
  // multiple loop-backs at the same row don't overlap exactly.
  const dip = 46 + edgeIndex * 22;
  const bx1 = fromPos.x + NODE_W / 2;
  const by1 = fromPos.y + NODE_H;
  const bx2 = toPos.x + NODE_W / 2;
  const by2 = toPos.y + NODE_H;
  const dipY = Math.max(by1, by2) + dip;
  return `M ${bx1} ${by1} C ${bx1} ${dipY}, ${bx2} ${dipY}, ${bx2} ${by2}`;
}
