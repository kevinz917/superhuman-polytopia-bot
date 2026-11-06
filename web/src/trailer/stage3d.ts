// The 3D layer over the Pixi board: an orthographic camera that lines up exactly with the board's isometric
// projection, two rigged right hands (the WebXR generic hand, MIT, Amazon) posed palm-down by forward kinematics,
// their shadows on the board, and the slab the board becomes while it flips.
//
// Scene units: one tile is T across, X runs along the board's columns and Z along its rows, Y is up. With
// T = HW·K·√2 and the camera pitched asin(0.6) and turned 45°, a ground point lands on the same pixel Pixi
// draws it on, and one unit of height rises 0.8 px on screen.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

export interface Projection {
  W: number;
  H: number;
  K: number; // Pixi world -> screen scale
  OX: number;
  OY: number;
  HW: number;
  HH: number;
}
export const SIN = 0.6,
  COS = 0.8;

// ---------- hand rig ----------
// Joint indices in the glb (all joints are siblings, posed in armature space).
const THUMB = [1, 2, 3, 4],
  FINGERS = [
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24],
  ];
// Model space: the palm faces -X, the fingers point -Y, the thumb is on -Z.
const PALM = new THREE.Vector3(-1, 0, 0);

export interface HandPose {
  grip: THREE.Vector3; // where the pinch point (between thumb and index tips) goes, scene units
  toward: THREE.Vector3; // horizontal direction the fingers point
  pitch: number; // radians the fingers tilt down from horizontal
  roll: number; // radians of pronation about the finger direction
  pinch: number; // 0 relaxed open, 1 closed on a piece
  curl: number; // extra curl of the ring and little fingers (0..1)
  scale: number; // scene units per metre
  alpha: number;
}

interface Rest {
  p: THREE.Vector3;
  q: THREE.Quaternion;
}

class Hand {
  root = new THREE.Group();
  private bones: THREE.Bone[] = [];
  private rest: Rest[] = [];
  private skin: THREE.MeshStandardMaterial;
  private uniforms = { rimColor: { value: new THREE.Color() }, fade: { value: 1 } };

  constructor(source: THREE.Object3D, rim: number) {
    const model = source.clone(true);
    let mesh: THREE.SkinnedMesh | null = null;
    model.traverse((o) => {
      if ((o as THREE.SkinnedMesh).isSkinnedMesh) mesh = o as THREE.SkinnedMesh;
    });
    if (!mesh) throw new Error("hand mesh missing");
    const m = mesh as THREE.SkinnedMesh;
    // The clone shares the source skeleton; give it its own bones from the cloned tree.
    const byName = new Map<string, THREE.Bone>();
    model.traverse((o) => {
      if ((o as THREE.Bone).isBone) byName.set(o.name, o as THREE.Bone);
    });
    const bones = m.skeleton.bones.map((b) => byName.get(b.name)!);
    m.bind(new THREE.Skeleton(bones, m.skeleton.boneInverses), m.bindMatrix);
    this.bones = bones;
    this.rest = bones.map((b) => ({ p: b.position.clone(), q: b.quaternion.clone() }));
    this.uniforms.rimColor.value.set(rim);

    // Skin: a pale, slightly warm, semi-translucent surface with a bright fresnel rim; the wrist fades out.
    this.skin = new THREE.MeshStandardMaterial({
      color: 0xf0cdb4,
      roughness: 0.5,
      metalness: 0,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
      depthFunc: THREE.LessEqualDepth,
    });
    this.skin.onBeforeCompile = (shader) => {
      shader.uniforms.rimColor = this.uniforms.rimColor;
      shader.uniforms.fade = this.uniforms.fade;
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying float vAlong;")
        .replace("#include <begin_vertex>", "#include <begin_vertex>\nvAlong = position.y;");
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", "#include <common>\nvarying float vAlong;\nuniform vec3 rimColor;\nuniform float fade;")
        .replace(
          "#include <dithering_fragment>",
          `#include <dithering_fragment>
          vec3 viewDir = isOrthographic ? vec3(0.0, 0.0, 1.0) : normalize(vViewPosition);
          float fr = pow(1.0 - clamp(abs(dot(normal, viewDir)), 0.0, 1.0), 2.2);
          gl_FragColor.rgb += rimColor * fr * 0.9;
          gl_FragColor.a = mix(gl_FragColor.a, 0.95, fr * 0.7);
          // The forearm dissolves a few centimetres past the wrist.
          gl_FragColor.a *= (1.0 - smoothstep(0.06, 0.125, vAlong)) * fade;`,
        );
    };
    m.material = this.skin;
    m.renderOrder = 2;
    m.castShadow = true;
    m.frustumCulled = false;
    // Depth pre-pass so only the nearest surface of the translucent hand shows.
    const pre = new THREE.SkinnedMesh(m.geometry, new THREE.MeshBasicMaterial({ colorWrite: false }));
    pre.bind(m.skeleton, m.bindMatrix);
    pre.renderOrder = 1;
    pre.frustumCulled = false;
    m.parent!.add(pre);
    // A short forearm out of the wrist toward the shoulder (model +Y), fitted to the wrist's cross-section
    // (centre x 0.039, z 0.010; half-widths 0.019 x 0.026 m) and fading out within a few centimetres.
    const arm = new THREE.CylinderGeometry(1, 1.08, 0.09, 32, 6, true);
    arm.translate(0, 0.105, 0);
    arm.scale(0.0185, 1, 0.0255);
    arm.translate(0.039, 0, 0.0102);
    const forearm = new THREE.Mesh(arm, this.skin);
    forearm.renderOrder = 2;
    const armPre = new THREE.Mesh(arm, pre.material);
    armPre.renderOrder = 1;
    model.add(forearm, armPre);
    this.root.add(model);
  }

  // Forward kinematics: rotate every joint downstream of `pivot` (and the pivot's own bone) about an axis.
  private turn(P: THREE.Vector3[], Q: THREE.Quaternion[], chain: number[], from: number, axis: THREE.Vector3, angle: number) {
    if (!angle) return;
    const r = new THREE.Quaternion().setFromAxisAngle(axis, angle),
      pivot = P[chain[from]].clone();
    for (let k = from; k < chain.length; k++) {
      const j = chain[k];
      if (k > from) P[j].sub(pivot).applyQuaternion(r).add(pivot);
      Q[j].premultiply(r);
    }
  }

  setRim(color: number) {
    this.uniforms.rimColor.value.set(color);
  }

  pose(h: HandPose) {
    const P = this.rest.map((r) => r.p.clone()),
      Q = this.rest.map((r) => r.q.clone());
    const pinch = h.pinch,
      relax = 0.18; // a hand at rest is never flat
    // Fingers: spread, then curl at the knuckle, middle and end joints toward the palm.
    const flex = [
      // [MCP, PIP, DIP] in degrees, open then pinched
      [[14, 18, 8], [44, 46, 22]],
      [[16, 22, 10], [52, 58, 26]],
      [[18, 26, 12], [62, 70, 34]],
      [[20, 28, 14], [66, 74, 38]],
    ];
    const spread = [[-7, -2], [-1, 0], [5, 2], [11, 4]];
    FINGERS.forEach((chain, f) => {
      const dir = P[chain[2]].clone().sub(P[chain[1]]).normalize(),
        lateral = dir.clone().cross(PALM).normalize();
      const u = Math.min(1, pinch + relax * (1 - pinch) + (f >= 2 ? h.curl * 0.6 : 0));
      this.turn(P, Q, chain, 1, PALM, THREE.MathUtils.degToRad(THREE.MathUtils.lerp(spread[f][0], spread[f][1], pinch)));
      const lat = lateral.clone().applyAxisAngle(PALM, THREE.MathUtils.degToRad(THREE.MathUtils.lerp(spread[f][0], spread[f][1], pinch)));
      for (let k = 0; k < 3; k++)
        this.turn(P, Q, chain, k + 1, lat, THREE.MathUtils.degToRad(THREE.MathUtils.lerp(flex[f][0][k], flex[f][1][k], u)));
    });
    // Thumb: swing across under the palm toward the index tip, then bend its two joints.
    {
      const base = P[THUMB[0]],
        tdir = P[THUMB[2]].clone().sub(P[THUMB[1]]).normalize(),
        lateral = tdir.clone().cross(PALM).normalize();
      void base;
      const swing = THREE.MathUtils.degToRad(THREE.MathUtils.lerp(8, 30, pinch));
      this.turn(P, Q, THUMB, 0, new THREE.Vector3(0, 0, 1).cross(PALM).normalize().negate(), swing * 0.5);
      this.turn(P, Q, THUMB, 0, PALM.clone().negate(), swing);
      const lat = lateral.clone();
      this.turn(P, Q, THUMB, 1, lat, THREE.MathUtils.degToRad(THREE.MathUtils.lerp(8, 22, pinch)));
      this.turn(P, Q, THUMB, 2, lat, THREE.MathUtils.degToRad(THREE.MathUtils.lerp(10, 30, pinch)));
    }
    this.bones.forEach((b, i) => {
      b.position.copy(P[i]);
      b.quaternion.copy(Q[i]);
    });
    // Pinch point in model space.
    const anchor = P[THUMB[3]].clone().add(P[FINGERS[0][4]]).multiplyScalar(0.5);
    // Model -> scene: back of the hand (+X) up, fingers (-Y) along `toward` tilted down by `pitch`.
    const fwd = h.toward.clone().setY(0).normalize(),
      f = fwd.clone().multiplyScalar(Math.cos(h.pitch)).add(new THREE.Vector3(0, -Math.sin(h.pitch), 0)),
      up = fwd.clone().multiplyScalar(Math.sin(h.pitch)).add(new THREE.Vector3(0, Math.cos(h.pitch), 0));
    const ex = up.clone().applyAxisAngle(f, h.roll),
      ey = f.clone().negate(),
      ez = ex.clone().cross(ey);
    const basis = new THREE.Matrix4().makeBasis(ex, ey, ez);
    this.root.quaternion.setFromRotationMatrix(basis);
    this.root.scale.setScalar(h.scale);
    const offset = anchor.clone().applyQuaternion(this.root.quaternion).multiplyScalar(h.scale);
    this.root.position.copy(h.grip).sub(offset);
    this.uniforms.fade.value = h.alpha;
    this.root.visible = h.alpha > 0.01;
  }
}

// ---------- stage ----------
export class Stage {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.OrthographicCamera;
  hands: Hand[] = [];
  T: number;
  private proj: Projection;
  slab = new THREE.Group();
  private faces: THREE.Mesh[] = [];
  private textures: THREE.CanvasTexture[] = [];
  private shadowPlane: THREE.Mesh;

  constructor(canvas: HTMLCanvasElement, proj: Projection, size: number) {
    this.proj = proj;
    const { W, H, K, HW, HH, OY } = proj;
    this.T = HW * K * Math.SQRT2;
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(W, H, false);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.camera = new THREE.OrthographicCamera(-W / 2, W / 2, H / 2, -H / 2, 1, 8000);
    const c0 = (H / 2 - OY) / (2 * HH * K),
      target = new THREE.Vector3(c0 * this.T, 0, c0 * this.T);
    const F = new THREE.Vector3(-COS / Math.SQRT2, -SIN, -COS / Math.SQRT2),
      U = new THREE.Vector3(-SIN / Math.SQRT2, COS, -SIN / Math.SQRT2);
    this.camera.position.copy(target).addScaledVector(F, -4000);
    this.camera.up.copy(U);
    this.camera.lookAt(target);
    this.camera.updateMatrixWorld();

    this.scene.add(new THREE.HemisphereLight(0xf4f7ff, 0x6b5b52, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    const mid = ((size - 1) / 2) * this.T;
    key.position.set(mid - 900, 1800, mid - 300);
    key.target.position.set(mid, 0, mid);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    const ext = size * this.T;
    Object.assign(key.shadow.camera, { left: -ext, right: ext, top: ext, bottom: -ext, near: 10, far: 6000 });
    key.shadow.radius = 6;
    key.shadow.bias = -0.0005;
    this.scene.add(key, key.target);
    const back = new THREE.DirectionalLight(0xcfe3ff, 0.9);
    back.position.set(mid + 600, 300, mid + 900);
    this.scene.add(back);

    this.shadowPlane = new THREE.Mesh(new THREE.PlaneGeometry(size * this.T * 1.6, size * this.T * 1.6), new THREE.ShadowMaterial({ opacity: 0.3 }));
    this.shadowPlane.rotation.x = -Math.PI / 2;
    this.shadowPlane.position.set(mid, 0.5, mid);
    this.shadowPlane.receiveShadow = true;
    this.scene.add(this.shadowPlane);
    this.scene.add(this.slab);
    this.slab.visible = false;
  }

  async loadHands(url: string, rims: number[]) {
    const gltf = await new GLTFLoader().loadAsync(url);
    for (const rim of rims) {
      const h = new Hand(gltf.scene, rim);
      this.hands.push(h);
      this.scene.add(h.root);
    }
  }

  // Pixi world (x, y) and a height in scene units -> scene point.
  ground(x: number, y: number, height = 0) {
    const { HW, HH } = this.proj;
    const a = x / HW,
      b = y / HH; // a = c - r, b = c + r
    return new THREE.Vector3(((a + b) / 2) * this.T, height, ((b - a) / 2) * this.T);
  }

  // ---------- the board as a slab, for the flip ----------
  // Front and back are the two board images (old map, new map), each mapped onto the ground plane so that,
  // lying flat, they cover exactly the pixels Pixi drew. The slab is SLAB thick and turns about its middle;
  // the back is built in its landed pose and turned half over, so after the 180° flip it lands pixel-exact too.
  static SLAB = 26;
  private box: THREE.Mesh | null = null;

  slabMatrix(angle: number, axis: THREE.Vector3, centre: THREE.Vector3, rise: number) {
    const y = -Stage.SLAB / 2;
    return new THREE.Matrix4()
      .makeTranslation(centre.x, y + rise, centre.z)
      .multiply(new THREE.Matrix4().makeRotationAxis(axis, angle))
      .multiply(new THREE.Matrix4().makeTranslation(-centre.x, -y, -centre.z));
  }

  // spin: how far the board has turned about its vertical axis (radians), so the body turns with it.
  setSlab(front: HTMLCanvasElement, back: HTMLCanvasElement, size: number, axis: THREE.Vector3, centre: THREE.Vector3, spin = 0) {
    for (const f of this.faces) {
      this.slab.remove(f);
      f.geometry.dispose();
    }
    for (const t of this.textures) t.dispose();
    this.faces = [];
    this.textures = [front, back].map((c) => {
      const t = new THREE.CanvasTexture(c);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 4;
      return t;
    });
    const { W, H } = this.proj,
      ext = size * this.T * 0.95;
    const decal = (texture: THREE.Texture, flipped: boolean, lift: number) => {
      const g = new THREE.PlaneGeometry(ext * 2, ext * 2, 1, 1);
      g.rotateX(-Math.PI / 2);
      g.translate(centre.x, lift, centre.z);
      const pos = g.attributes.position as THREE.BufferAttribute,
        uv = g.attributes.uv as THREE.BufferAttribute,
        v = new THREE.Vector3();
      for (let i = 0; i < pos.count; i++) {
        v.fromBufferAttribute(pos, i).project(this.camera);
        uv.setXY(i, (v.x + 1) / 2, (v.y + 1) / 2);
        void W;
        void H;
      }
      // Built landed; turn it back half over so the flip brings it home. Its face then points down.
      if (flipped) g.applyMatrix4(this.slabMatrix(-Math.PI, axis, centre, 0));
      const mesh = new THREE.Mesh(
        g,
        new THREE.MeshBasicMaterial({ map: texture, transparent: true, side: THREE.FrontSide, depthWrite: false }),
      );
      mesh.renderOrder = 0;
      this.slab.add(mesh);
      this.faces.push(mesh);
    };
    decal(this.textures[0], false, 0.6);
    decal(this.textures[1], true, 0.6);
    if (!this.box) {
      // The board's body: earth sides in the light, so the turn reads as a solid board.
      this.box = new THREE.Mesh(
        new THREE.BoxGeometry(size * this.T, Stage.SLAB, size * this.T),
        new THREE.MeshStandardMaterial({ color: 0x7a4a2a, roughness: 0.9 }),
      );
      const mid = ((size - 1) / 2) * this.T;
      this.box.position.set(mid, -Stage.SLAB / 2 - 0.2, mid);
      this.slab.add(this.box);
    }
    this.box.rotation.y = spin;
  }

  setSlabPose(angle: number, axis: THREE.Vector3, centre: THREE.Vector3, rise: number) {
    this.slab.visible = true;
    this.slab.position.set(0, 0, 0);
    this.slab.quaternion.identity();
    this.slab.scale.set(1, 1, 1);
    this.slab.applyMatrix4(this.slabMatrix(angle, axis, centre, rise));
  }

  hideSlab() {
    this.slab.visible = false;
  }

  render(shake: [number, number] = [0, 0]) {
    const { W, H } = this.proj;
    this.camera.left = -W / 2 - shake[0];
    this.camera.right = W / 2 - shake[0];
    this.camera.top = H / 2 + shake[1];
    this.camera.bottom = -H / 2 + shake[1];
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
  }
}
