import * as THREE from './vendor/three/three.module.js';
import { OrbitControls } from './vendor/three/OrbitControls.js';

export function parsePreview(buffer) {
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== 'S3MP') throw new Error('Bad preview data');
  const dv = new DataView(buffer);
  const nv = dv.getUint32(8, true);
  const nf = dv.getUint32(12, true);
  const nr = dv.getUint32(16, true);
  const nn = dv.getUint32(20, true);
  let off = 24;
  const positions = new Float32Array(buffer, off, nv * 3); off += nv * 12;
  const indices = new Uint32Array(buffer, off, nf * 3); off += nf * 12;
  const rim = new Float32Array(buffer, off, nr * 6); off += nr * 24;
  const nm = new Float32Array(buffer, off, nn * 6);
  return { positions, indices, rim, nm, nv, nf };
}

function lines(data, color) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(data, 3));
  const mat = new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 });
  const obj = new THREE.LineSegments(geom, mat);
  obj.renderOrder = 3;
  obj.userData.defect = true;
  return obj;
}

// For an open scan the summed face normals point away from the unscanned side;
// for a closed mesh they cancel out and there is no preferred side.
function scannedSide(positions, indices) {
  let sx = 0, sy = 0, sz = 0, total = 0;
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
    const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    sx += nx; sy += ny; sz += nz;
    total += Math.hypot(nx, ny, nz);
  }
  const len = Math.hypot(sx, sy, sz);
  if (!total || len < 0.1 * total) return null;
  return new THREE.Vector3(sx / len, sy / len, sz / len);
}

const DEFAULT_VIEW = new THREE.Vector3(0.9, -1.35, 0.85).normalize();

function disposeTree(obj) {
  obj.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
  });
}

export class DualViewer {
  constructor(beforeEl, afterEl) {
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.01, 1e6);
    this.camera.up.set(0, 0, 1);
    this.showDefects = true;
    this.pending = false;
    this.panes = { before: this.makePane(beforeEl), after: this.makePane(afterEl) };
    this.fitted = false;
  }

  makePane(el) {
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x6d7a8c, 1.4));
    const light = new THREE.DirectionalLight(0xffffff, 1.8);
    scene.add(light);
    scene.add(light.target);
    const controls = new OrbitControls(this.camera, renderer.domElement);
    controls.enableDamping = false;
    controls.addEventListener('change', () => this.render());
    renderer.domElement.addEventListener('dblclick', () => this.fit());
    const pane = { el, renderer, scene, controls, light, group: null, size: null, scale: 1 };
    new ResizeObserver(() => this.resize()).observe(el);
    return pane;
  }

  setMesh(which, buffer, { scale = 1, refit = false } = {}) {
    const data = parsePreview(buffer);
    const pane = this.panes[which];
    this.clear(which, false);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
    geom.setIndex(new THREE.BufferAttribute(data.indices, 1));
    geom.computeVertexNormals();
    geom.computeBoundingBox();
    const box = geom.boundingBox;
    const centre = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    const inner = new THREE.Group();
    inner.position.copy(centre).multiplyScalar(-1);
    const front = new THREE.MeshStandardMaterial({
      color: 0xa9b8cc, roughness: 0.72, metalness: 0.02, side: THREE.FrontSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    const back = new THREE.MeshBasicMaterial({ color: 0xf08c3a, side: THREE.BackSide });
    inner.add(new THREE.Mesh(geom, front));
    inner.add(new THREE.Mesh(geom, back));
    if (data.rim.length) inner.add(lines(data.rim, 0xe0245e));
    if (data.nm.length) inner.add(lines(data.nm, 0x9b30ff));

    const extent = Math.max(size.x, size.y, size.z) || 1;
    const step = Math.pow(10, Math.floor(Math.log10(extent / 4)));
    const gridSize = Math.ceil((extent * 1.6) / step) * step;
    const grid = new THREE.GridHelper(gridSize, Math.min(200, Math.round(gridSize / step)), 0x9aa7ba, 0xd3dae5);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(centre.x, centre.y, box.min.z);
    grid.material.transparent = true;
    grid.material.opacity = 0.7;
    inner.add(grid);

    const group = new THREE.Group();
    group.add(inner);
    group.scale.setScalar(scale);
    pane.scene.add(group);
    pane.group = group;
    pane.size = size;
    pane.scale = scale;
    this.applyDefects();
    if (!this.fitted) {
      const side = scannedSide(data.positions, data.indices);
      if (side) {
        const tilt = DEFAULT_VIEW.clone().sub(side.clone().multiplyScalar(DEFAULT_VIEW.dot(side)));
        const dir = side.clone().add(tilt.multiplyScalar(0.6)).normalize();
        if (Math.abs(dir.z) > 0.92) dir.add(new THREE.Vector3(0, -0.5, 0)).normalize();
        this.viewDir = dir;
      } else {
        this.viewDir = DEFAULT_VIEW.clone();
      }
    }
    if (refit || !this.fitted) this.fit();
    else this.render();
    return { triangles: data.nf, rimEdges: data.rim.length / 6, nmEdges: data.nm.length / 6 };
  }

  setScale(which, scale) {
    const pane = this.panes[which];
    if (!pane.group) return;
    pane.scale = scale;
    pane.group.scale.setScalar(scale);
    this.fit();
  }

  clear(which, render = true) {
    const pane = this.panes[which];
    if (pane.group) {
      pane.scene.remove(pane.group);
      disposeTree(pane.group);
      pane.group = null;
      pane.size = null;
    }
    if (render) this.render();
  }

  setShowDefects(show) {
    this.showDefects = show;
    this.applyDefects();
    this.render();
  }

  applyDefects() {
    for (const pane of Object.values(this.panes)) {
      if (!pane.group) continue;
      pane.group.traverse((o) => { if (o.userData.defect) o.visible = this.showDefects; });
    }
  }

  fit() {
    let radius = 0;
    for (const pane of Object.values(this.panes)) {
      if (pane.size) radius = Math.max(radius, (pane.size.length() / 2) * pane.scale);
    }
    if (!radius) return;
    const fov = (this.camera.fov * Math.PI) / 180;
    const dist = (radius / Math.sin(fov / 2)) * 1.08;
    const dir = (this.viewDir || DEFAULT_VIEW).clone();
    this.camera.position.copy(dir.multiplyScalar(dist));
    this.camera.near = Math.max(dist / 1000, 1e-4);
    this.camera.far = dist * 100;
    this.camera.updateProjectionMatrix();
    for (const pane of Object.values(this.panes)) {
      pane.controls.target.set(0, 0, 0);
      pane.controls.update();
    }
    this.fitted = true;
    this.render();
  }

  resize() {
    for (const pane of Object.values(this.panes)) {
      const w = pane.el.clientWidth;
      const h = pane.el.clientHeight;
      if (w && h) {
        pane.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
      }
    }
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      for (const pane of Object.values(this.panes)) {
        pane.light.position.copy(this.camera.position);
        pane.renderer.render(pane.scene, this.camera);
      }
    });
  }
}
