import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { Vector3, type Group } from "three";
import { particleColors } from "../../design/tokens";
import { sphereLayout } from "../layout";
import { BLUEGOLD_CONFIG, BLUEGOLD_LOOK, createParticleSystem } from "./engine";
import { usePointerTarget } from "./usePointerTarget";

export interface ParticleSphereProps {
  particles: number;
  reducedMotion: boolean;
  /** Diagnostics only (?scene=fail). */
  broken?: boolean;
}

const ROTATION_SPEED = 0.06; // rad/s
const TILT: [number, number, number] = [0.35, 0, 0.08];

/**
 * One THREE.Points object (one draw call). Buffers are built once per
 * particle count; each frame writes only a handful of uniforms and one
 * rotation. No per-particle JavaScript runs per frame.
 */
export function ParticleSphere({ particles, reducedMotion, broken = false }: ParticleSphereProps) {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);
  const group = useRef<Group>(null);

  const system = useMemo(
    () =>
      createParticleSystem({
        config: { ...BLUEGOLD_CONFIG, count: particles },
        look: BLUEGOLD_LOOK,
        colors: particleColors,
        pixelRatio: gl.getPixelRatio(),
        motion: reducedMotion ? 0 : 1,
        broken,
      }),
    // reducedMotion is applied per frame via uMotion; it must not rebuild buffers.
    [particles, broken, gl],
  );
  useEffect(() => () => system.dispose(), [system]);

  // Re-render once when the motion preference changes (needed in "demand" mode).
  useEffect(() => invalidate(), [reducedMotion, invalidate]);

  const pointer = usePointerTarget(!reducedMotion, gl.domElement);
  const scratch = useMemo(() => ({ ndc: new Vector3(), dir: new Vector3(), hit: new Vector3() }), []);
  const layout = sphereLayout(size.width, size.height);

  useFrame((state, delta) => {
    const u = system.uniforms;
    const dt = Math.min(delta, 0.05); // no jump after a pause
    const motion = reducedMotion ? 0 : 1;
    u.uMotion.value = motion;
    u.uTime.value += dt;
    u.uPixelRatio.value = state.viewport.dpr;
    u.uSize.value = BLUEGOLD_LOOK.pointSize * layout.pointScale;

    // Pointer -> world point on the plane z = 0 (one ray, not per particle).
    let targetStrength = 0;
    const p = pointer.current;
    if (motion === 1 && p.active) {
      scratch.ndc.set(p.ndcX, p.ndcY, 0.5).unproject(camera);
      scratch.dir.copy(scratch.ndc).sub(camera.position).normalize();
      if (Math.abs(scratch.dir.z) > 1e-6) {
        const t = -camera.position.z / scratch.dir.z;
        scratch.hit.copy(camera.position).addScaledVector(scratch.dir, t);
        u.uPointer.value.lerp(scratch.hit, 1 - Math.exp(-dt * 12));
        targetStrength = 1;
      }
    }
    u.uPointerStrength.value +=
      (targetStrength - u.uPointerStrength.value) * (1 - Math.exp(-dt * 4));

    if (group.current) group.current.rotation.y += dt * ROTATION_SPEED * motion;
  });

  // Outer group: layout + fixed tilt (props, may re-render). Inner group: spin
  // only, mutated in useFrame, so a re-render never resets the rotation.
  return (
    <group position={layout.position} scale={layout.scale} rotation={TILT}>
      <group ref={group}>
        <points geometry={system.geometry} material={system.material} />
      </group>
    </group>
  );
}
