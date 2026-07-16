/**
 * Blender-style orientation gizmo. A small SVG overlay in the viewport corner showing the world
 * X/Y/Z axes as the CURRENT camera sees them; click a +/- axis ball to snap to that view. Reads the
 * player's exact orbit basis (mirrors packages/core/src/camera.ts lookAt), so it always matches the
 * render. Camera-only — never touches edits/ranges; works in and out of edit mode.
 */
const AX = [
  { k: 0, col: "#e0605f", label: "X" },   // conventional RGB axes, toned to the app's muted palette
  { k: 1, col: "#7ec36a", label: "Y" },
  { k: 2, col: "#5f92e0", label: "Z" },
];
const ELEV_CLAMP = 1.4;   // matches the player's drag clamp (±~80°); a literal ±90° would jump on next drag
const NS = "http://www.w3.org/2000/svg";
const C = 34, R = 22, BALL = 6.5;   // viewBox 0..68, center, axis length, ball radius

export function initGizmo() {
  const el = document.getElementById("axisGizmo");
  if (!el) return;
  const bg = document.createElementNS(NS, "circle");   // faint backdrop so the axes read over any model
  bg.setAttribute("cx", C); bg.setAttribute("cy", C); bg.setAttribute("r", "32"); bg.setAttribute("fill", "rgba(18,20,26,0.34)");
  el.append(bg);
  const linesG = document.createElementNS(NS, "g");
  const ballsG = document.createElementNS(NS, "g");
  el.append(linesG, ballsG);

  const mkBall = (a, sign) => {
    const g = document.createElementNS(NS, "g"); g.style.cursor = "pointer";
    const c = document.createElementNS(NS, "circle"); c.setAttribute("r", String(BALL));
    const t = document.createElementNS(NS, "text"); t.setAttribute("text-anchor", "middle"); t.setAttribute("dominant-baseline", "central");
    t.setAttribute("font-size", "8"); t.setAttribute("font-weight", "700"); t.setAttribute("fill", "#14161b"); t.style.pointerEvents = "none";
    t.textContent = sign > 0 ? a.label : "";
    g.append(c, t);
    g.onclick = () => snap(a.k, sign);
    const title = document.createElementNS(NS, "title"); title.textContent = `look ${sign > 0 ? "+" : "−"}${a.label}`; g.append(title);
    ballsG.append(g);
    return { g, c, t, sign, col: a.col };
  };
  const objs = AX.map((a) => {
    const line = document.createElementNS(NS, "line");
    line.setAttribute("stroke", a.col); line.setAttribute("stroke-width", "2"); line.setAttribute("stroke-linecap", "round");
    linesG.append(line);
    return { line, pos: mkBall(a, 1), neg: mkBall(a, -1), col: a.col };
  });

  function snap(k, sign) {
    const p = window.__ares; if (!p) return;
    const dir = [0, 0, 0]; dir[k] = sign;
    const elev = Math.max(-ELEV_CLAMP, Math.min(ELEV_CLAMP, Math.asin(dir[1])));
    const az = Math.atan2(dir[0], dir[2]);
    const cam = p.getCamera();
    p.setCamera({ azimuth: az, elevation: elev, distance: cam.distance, target: cam.target });
  }

  function place(ball, px, py, near) {
    ball.c.setAttribute("cx", px); ball.c.setAttribute("cy", py);
    ball.t.setAttribute("x", px); ball.t.setAttribute("y", py);
    if (ball.sign > 0) {   // + end: filled + labeled; dim when facing away
      ball.c.setAttribute("fill", ball.col); ball.c.setAttribute("stroke", "none");
      ball.c.setAttribute("opacity", near ? "1" : "0.5"); ball.t.setAttribute("opacity", near ? "1" : "0.5");
    } else {               // − end: hollow ring
      ball.c.setAttribute("fill", near ? ball.col : "#181a20"); ball.c.setAttribute("stroke", ball.col); ball.c.setAttribute("stroke-width", "1.5");
      ball.c.setAttribute("opacity", near ? "0.9" : "0.85");
    }
  }
  function draw(az, elev) {
    const ce = Math.cos(elev), se = Math.sin(elev);
    const z = [ce * Math.sin(az), se, ce * Math.cos(az)];               // view dir (target→eye), = lookAt z basis
    const rl = Math.hypot(z[2], z[0]) || 1;
    const right = [z[2] / rl, 0, -z[0] / rl];                           // = normalize(worldUp × z)
    const up = [z[1] * right[2] - z[2] * right[1], z[2] * right[0] - z[0] * right[2], z[0] * right[1] - z[1] * right[0]]; // z × right
    const ends = [];
    objs.forEach((o, i) => {
      const k = AX[i].k;
      const sx = right[k], sy = up[k], depth = z[k];                    // world axis k projects to (right·e, up·e, z·e)
      o.line.setAttribute("x1", C); o.line.setAttribute("y1", C);
      o.line.setAttribute("x2", C + R * sx); o.line.setAttribute("y2", C - R * sy);
      place(o.pos, C + R * sx, C - R * sy, depth >= 0);
      place(o.neg, C - R * sx, C + R * sy, -depth >= 0);
      ends.push({ g: o.pos.g, d: depth }, { g: o.neg.g, d: -depth });
    });
    ends.sort((a, b) => a.d - b.d).forEach((e) => ballsG.append(e.g));  // far balls painted first (behind)
  }

  let lastAz = NaN, lastEl = NaN, lastRight = NaN;
  (function frame() {
    const p = window.__ares;
    if (p) {
      const cam = p.getCamera();
      if (cam.azimuth !== lastAz || cam.elevation !== lastEl) { lastAz = cam.azimuth; lastEl = cam.elevation; draw(cam.azimuth, cam.elevation); }
    }
    // keep the gizmo just left of the edit rail when it's open (drag-resizable), else in the corner
    const ep = document.getElementById("editPanel");
    const off = (ep && ep.style.display !== "none") ? (ep.offsetWidth + 12) : 12;
    if (off !== lastRight) { el.style.right = off + "px"; lastRight = off; }
    requestAnimationFrame(frame);
  })();
}
