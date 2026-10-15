/*
 * Lunar Lander, ported line-for-line from Gymnasium's lunar_lander.py (v3) to planck.js
 * (a JavaScript port of Box2D). Same constants, bodies, joints, collision filters,
 * engine impulse model, wind/turbulence model, observation and reward.
 *
 * Works in the browser (expects global `planck`) and in Node (for parity tests).
 */
(function (root) {
  "use strict";
  const pl = root.planck;
  const Vec2 = pl.Vec2;

  const FPS = 50, SCALE = 30.0;
  const MAIN_ENGINE_POWER = 13.0, SIDE_ENGINE_POWER = 0.6;
  const INITIAL_RANDOM = 1000.0;
  const LANDER_POLY = [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]];
  const LEG_AWAY = 20, LEG_DOWN = 18, LEG_W = 2, LEG_H = 8, LEG_SPRING_TORQUE = 40;
  const SIDE_ENGINE_HEIGHT = 14, SIDE_ENGINE_AWAY = 12, MAIN_ENGINE_Y_LOCATION = 4;
  const VIEWPORT_W = 600, VIEWPORT_H = 400;
  const W = VIEWPORT_W / SCALE, H = VIEWPORT_H / SCALE;
  const CHUNKS = 11;

  // Small seeded PRNG (mulberry32) so episodes are reproducible.
  function makeRng(seed) {
    let a = seed >>> 0;
    const next = () => {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return { uniform: (lo, hi) => lo + (hi - lo) * next(), int: (lo, hi) => Math.floor(lo + (hi - lo) * next()) };
  }

  class LunarLander {
    constructor(opts = {}) {
      this.continuous = !!opts.continuous;
      this.gravity = opts.gravity ?? -10.0;
      this.enableWind = !!opts.enableWind;
      this.windPower = opts.windPower ?? 15.0;
      this.turbulencePower = opts.turbulencePower ?? 1.5;
      this.maxSteps = opts.maxSteps ?? 1000;
    }

    reset(seed) {
      this.rng = makeRng(seed ?? Math.floor(Math.random() * 2 ** 31));
      const world = (this.world = new pl.World({ gravity: Vec2(0, this.gravity) }));
      this.gameOver = false;
      this.prevShaping = null;
      this.t = 0;
      this.particles = [];
      this.windMag = 0;
      this.lastAction = 0;

      // terrain
      const height = Array.from({ length: CHUNKS + 1 }, () => this.rng.uniform(0, H / 2));
      const chunkX = Array.from({ length: CHUNKS }, (_, i) => (W / (CHUNKS - 1)) * i);
      this.helipadX1 = chunkX[CHUNKS / 2 - 1 | 0];
      this.helipadX2 = chunkX[(CHUNKS / 2 | 0) + 1];
      this.helipadY = H / 4;
      const mid = CHUNKS / 2 | 0;
      for (const k of [-2, -1, 0, 1, 2]) height[mid + k] = this.helipadY;
      const smoothY = Array.from({ length: CHUNKS }, (_, i) =>
        0.33 * ((i === 0 ? height[CHUNKS] : height[i - 1]) + height[i] + height[i + 1]));
      // (python's height[-1] for i=0 wraps to the last element — reproduced above)

      this.moon = world.createBody();
      this.moon.createFixture(pl.Edge(Vec2(0, 0), Vec2(W, 0)), { density: 0 });
      this.terrain = [];
      for (let i = 0; i < CHUNKS - 1; i++) {
        const p1 = [chunkX[i], smoothY[i]], p2 = [chunkX[i + 1], smoothY[i + 1]];
        this.moon.createFixture(pl.Edge(Vec2(...p1), Vec2(...p2)), { density: 0, friction: 0.1 });
        this.terrain.push([p1, p2]);
      }

      // lander
      const ix = W / 2, iy = H;
      this.lander = world.createDynamicBody({ position: Vec2(ix, iy), angle: 0 });
      this.lander.createFixture(pl.Polygon(LANDER_POLY.map(([x, y]) => Vec2(x / SCALE, y / SCALE))), {
        density: 5.0, friction: 0.1, filterCategoryBits: 0x0010, filterMaskBits: 0x001, restitution: 0.0,
      });
      this.lander.applyForceToCenter(
        Vec2(this.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), this.rng.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), true);

      if (this.enableWind) {
        this.windIdx = this.rng.int(-9999, 9999);
        this.torqueIdx = this.rng.int(-9999, 9999);
      }

      // legs
      this.legs = [];
      this.legContact = [false, false];
      for (const i of [-1, 1]) {
        const leg = world.createDynamicBody({ position: Vec2(ix - (i * LEG_AWAY) / SCALE, iy), angle: i * 0.05 });
        leg.createFixture(pl.Box(LEG_W / SCALE, LEG_H / SCALE), {
          density: 1.0, restitution: 0.0, filterCategoryBits: 0x0020, filterMaskBits: 0x001,
        });
        const jd = {
          bodyA: this.lander, bodyB: leg,
          localAnchorA: Vec2(0, 0), localAnchorB: Vec2((i * LEG_AWAY) / SCALE, LEG_DOWN / SCALE),
          referenceAngle: 0, // pybox2d default; planck would otherwise infer it from body angles
          enableMotor: true, enableLimit: true, maxMotorTorque: LEG_SPRING_TORQUE, motorSpeed: 0.3 * i,
          lowerAngle: i === -1 ? 0.9 - 0.5 : -0.9, upperAngle: i === -1 ? 0.9 : -0.9 + 0.5,
        };
        world.createJoint(new pl.RevoluteJoint(jd));
        this.legs.push(leg);
      }

      const isLeg = (b) => this.legs.indexOf(b);
      world.on("begin-contact", (c) => {
        const a = c.getFixtureA().getBody(), b = c.getFixtureB().getBody();
        if (a === this.lander || b === this.lander) this.gameOver = true;
        for (const body of [a, b]) { const k = isLeg(body); if (k >= 0) this.legContact[k] = true; }
      });
      world.on("end-contact", (c) => {
        const a = c.getFixtureA().getBody(), b = c.getFixtureB().getBody();
        for (const body of [a, b]) { const k = isLeg(body); if (k >= 0) this.legContact[k] = false; }
      });

      return this.step(this.continuous ? [0, 0] : 0).obs;
    }

    step(action) {
      const L = this.lander;
      this.windMag = 0;
      if (this.enableWind && !(this.legContact[0] || this.legContact[1])) {
        const windMag = Math.tanh(Math.sin(0.02 * this.windIdx) + Math.sin(Math.PI * 0.01 * this.windIdx)) * this.windPower;
        this.windIdx += 1;
        L.applyForceToCenter(Vec2(windMag, 0), true);
        const torqueMag = Math.tanh(Math.sin(0.02 * this.torqueIdx) + Math.sin(Math.PI * 0.01 * this.torqueIdx)) * this.turbulencePower;
        this.torqueIdx += 1;
        L.applyTorque(torqueMag, true);
        this.windMag = windMag;
      }
      if (this.continuous) action = action.map((v) => Math.max(-1, Math.min(1, v)));

      const ang = L.getAngle();
      const tip = [Math.sin(ang), Math.cos(ang)];
      const side = [-tip[1], tip[0]];
      const disp = [this.rng.uniform(-1, 1) / SCALE, this.rng.uniform(-1, 1) / SCALE];

      let mPower = 0;
      if ((this.continuous && action[0] > 0) || (!this.continuous && action === 2)) {
        mPower = this.continuous ? (Math.min(Math.max(action[0], 0), 1) + 1) * 0.5 : 1.0;
        const ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * disp[0]) + side[0] * disp[1];
        const oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * disp[0]) - side[1] * disp[1];
        const p = L.getPosition();
        const pos = Vec2(p.x + ox, p.y + oy);
        L.applyLinearImpulse(Vec2(-ox * MAIN_ENGINE_POWER * mPower, -oy * MAIN_ENGINE_POWER * mPower), pos, true);
        this.particles.push({ x: pos.x, y: pos.y, vx: ox * 40 * mPower, vy: oy * 40 * mPower, life: 1, big: true });
      }
      let sPower = 0, direction = 0;
      if ((this.continuous && Math.abs(action[1]) > 0.5) || (!this.continuous && (action === 1 || action === 3))) {
        if (this.continuous) { direction = Math.sign(action[1]); sPower = Math.min(Math.max(Math.abs(action[1]), 0.5), 1.0); }
        else { direction = action - 2; sPower = 1.0; }
        const ox = tip[0] * disp[0] + side[0] * (3 * disp[1] + (direction * SIDE_ENGINE_AWAY) / SCALE);
        const oy = -tip[1] * disp[0] - side[1] * (3 * disp[1] + (direction * SIDE_ENGINE_AWAY) / SCALE);
        const p = L.getPosition();
        const pos = Vec2(p.x + ox - (tip[0] * 17) / SCALE, p.y + oy + (tip[1] * SIDE_ENGINE_HEIGHT) / SCALE);
        L.applyLinearImpulse(Vec2(-ox * SIDE_ENGINE_POWER * sPower, -oy * SIDE_ENGINE_POWER * sPower), pos, true);
        this.particles.push({ x: pos.x, y: pos.y, vx: ox * 150, vy: oy * 150, life: 0.6, big: false });
      }
      this.mPower = mPower; this.sPower = sPower; this.sideDir = direction;

      this.world.step(1.0 / FPS, 6 * 30, 2 * 30);
      this.t += 1;

      const pos = L.getPosition(), vel = L.getLinearVelocity();
      const obs = [
        (pos.x - W / 2) / (W / 2),
        (pos.y - (this.helipadY + LEG_DOWN / SCALE)) / (H / 2),
        (vel.x * (W / 2)) / FPS,
        (vel.y * (H / 2)) / FPS,
        L.getAngle(),
        (20.0 * L.getAngularVelocity()) / FPS,
        this.legContact[0] ? 1 : 0,
        this.legContact[1] ? 1 : 0,
      ];
      const shaping = -100 * Math.hypot(obs[0], obs[1]) - 100 * Math.hypot(obs[2], obs[3]) - 100 * Math.abs(obs[4]) + 10 * obs[6] + 10 * obs[7];
      let reward = this.prevShaping === null ? 0 : shaping - this.prevShaping;
      this.prevShaping = shaping;
      reward -= mPower * 0.3 + sPower * 0.03;

      let terminated = false, outcome = null;
      if (this.gameOver || Math.abs(obs[0]) >= 1.0) { terminated = true; reward = -100; outcome = Math.abs(obs[0]) >= 1 ? "out_of_bounds" : "crashed"; }
      if (!L.isAwake()) { terminated = true; reward = 100; outcome = "landed"; }
      const truncated = !terminated && this.t >= this.maxSteps + 1; // +1: reset() takes one step, as TimeLimit counts it separately
      if (truncated) outcome = "timeout";

      for (const p of this.particles) { p.x += p.vx / FPS; p.y += p.vy / FPS; p.life -= 0.04; }
      this.particles = this.particles.filter((p) => p.life > 0);

      return { obs, reward, terminated, truncated, outcome };
    }
  }

  // Tiny MLP policy runner for weights exported by scripts/export_policy.py
  class MLPPolicy {
    constructor(spec) { this.spec = spec; }
    forward(obs) {
      const s = this.spec;
      let x = obs.map((v, i) => Math.max(-10, Math.min(10, (v - s.obs_mean[i]) / Math.sqrt(s.obs_var[i] + 1e-8))));
      const acts = [];
      s.layers.forEach((layer, li) => {
        const y = layer.b.slice();
        for (let o = 0; o < y.length; o++) { const row = layer.W[o]; let acc = y[o]; for (let k = 0; k < x.length; k++) acc += row[k] * x[k]; y[o] = acc; }
        x = li < s.layers.length - 1 ? y.map(Math.tanh) : y;
        acts.push(x);
      });
      return { out: x, activations: acts };
    }
    act(obs) {
      const { out, activations } = this.forward(obs);
      if (this.spec.continuous) return { action: out.map((v) => Math.max(-1, Math.min(1, v))), out, activations };
      let best = 0; for (let i = 1; i < out.length; i++) if (out[i] > out[best]) best = i;
      return { action: best, out, activations };
    }
  }

  root.LunarLanderJS = { LunarLander, MLPPolicy, consts: { W, H, SCALE, FPS, LANDER_POLY, LEG_W, LEG_H, VIEWPORT_W, VIEWPORT_H } };
})(typeof window !== "undefined" ? window : globalThis);
