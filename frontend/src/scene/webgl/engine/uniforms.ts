import { Color, Vector3 } from "three";
import type { ParticleLook } from "./config";

/**
 * Every uniform the particle shaders declare, with its GLSL type. The
 * shader-contract test checks this list against the GLSL sources both ways.
 */
export const UNIFORM_TYPES = {
  uTime: "float",
  uMotion: "float",
  uRadius: "float",
  uTurbulence: "float",
  uPointer: "vec3",
  uPointerStrength: "float",
  uPointerRadius: "float",
  uSize: "float",
  uPixelRatio: "float",
  uOpacity: "float",
  uColorBlue: "vec3",
  uColorGold: "vec3",
} as const;

export type UniformName = keyof typeof UNIFORM_TYPES;

export interface ParticleUniforms {
  uTime: { value: number };
  uMotion: { value: number };
  uRadius: { value: number };
  uTurbulence: { value: number };
  uPointer: { value: Vector3 };
  uPointerStrength: { value: number };
  uPointerRadius: { value: number };
  uSize: { value: number };
  uPixelRatio: { value: number };
  uOpacity: { value: number };
  uColorBlue: { value: Color };
  uColorGold: { value: Color };
}

export interface UniformInit {
  look: ParticleLook;
  /** sRGB hex colours; Color.set converts them to linear (three's working space). */
  blue: string;
  gold: string;
  pixelRatio: number;
  /** 1 = full motion, 0 = static (prefers-reduced-motion). */
  motion: number;
}

export function createUniforms(init: UniformInit): ParticleUniforms {
  return {
    uTime: { value: 0 },
    uMotion: { value: init.motion },
    uRadius: { value: init.look.radius },
    uTurbulence: { value: init.look.turbulence },
    uPointer: { value: new Vector3(0, 0, 0) },
    uPointerStrength: { value: 0 },
    uPointerRadius: { value: init.look.pointerRadius },
    uSize: { value: init.look.pointSize },
    uPixelRatio: { value: init.pixelRatio },
    uOpacity: { value: init.look.opacity },
    uColorBlue: { value: new Color(init.blue) },
    uColorGold: { value: new Color(init.gold) },
  };
}
