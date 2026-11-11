/**
 * jsdom cannot compile GLSL, so this checks the CPU <-> GPU contract instead:
 * every uniform/attribute/varying name and type agrees between the shader
 * sources and the JavaScript that feeds them. A mismatch here would compile
 * fine and silently render nothing.
 */
import { describe, expect, it } from "vitest";
import { createParticleGeometry } from "./geometry";
import { buildParticleAttributes } from "./attributes";
import { BLUEGOLD_CONFIG, BLUEGOLD_LOOK } from "./config";
import { FRAGMENT_SHADER, VERTEX_SHADER } from "./material";
import fragmentSource from "./shaders/particle.frag.glsl?raw";
import vertexSource from "./shaders/particle.vert.glsl?raw";
import { UNIFORM_TYPES, createUniforms } from "./uniforms";

type Decl = Map<string, string>;

function declarations(src: string, qualifier: "uniform" | "attribute" | "varying"): Decl {
  const map: Decl = new Map();
  const re = new RegExp(`^\\s*${qualifier}\\s+(\\w+)\\s+(\\w+)\\s*;`, "gm");
  for (const m of src.matchAll(re)) map.set(m[2]!, m[1]!);
  return map;
}

describe("shader contract", () => {
  const vertUniforms = declarations(vertexSource, "uniform");
  const fragUniforms = declarations(fragmentSource, "uniform");

  it("every declared uniform is in the schema, with the same type", () => {
    for (const [name, type] of [...vertUniforms, ...fragUniforms]) {
      expect(UNIFORM_TYPES, name).toHaveProperty(name);
      expect(UNIFORM_TYPES[name as keyof typeof UNIFORM_TYPES], name).toBe(type);
    }
  });

  it("every schema uniform is declared in a shader and created by createUniforms", () => {
    const declared = new Set([...vertUniforms.keys(), ...fragUniforms.keys()]);
    const created = createUniforms({
      look: BLUEGOLD_LOOK,
      blue: "#000000",
      gold: "#ffffff",
      pixelRatio: 1,
      motion: 1,
    });
    for (const name of Object.keys(UNIFORM_TYPES)) {
      expect(declared.has(name), name).toBe(true);
      expect(created, name).toHaveProperty(name);
    }
    expect(Object.keys(created).sort()).toEqual(Object.keys(UNIFORM_TYPES).sort());
  });

  it("uniform JS values match their GLSL types", () => {
    const u = createUniforms({ look: BLUEGOLD_LOOK, blue: "#000000", gold: "#ffffff", pixelRatio: 1, motion: 1 });
    for (const [name, type] of Object.entries(UNIFORM_TYPES)) {
      const value = (u as unknown as Record<string, { value: unknown }>)[name]!.value;
      if (type === "float") expect(typeof value, name).toBe("number");
      if (type === "vec3") {
        // three uploads both Vector3 (x,y,z) and Color (r,g,b) as a vec3.
        const v = value as Record<string, unknown>;
        const xyz = ["x", "y", "z"].every((k) => typeof v[k] === "number");
        const rgb = ["r", "g", "b"].every((k) => typeof v[k] === "number");
        expect(xyz || rgb, name).toBe(true);
      }
    }
  });

  it("custom attributes match the geometry's buffers and item sizes", () => {
    const attrs = declarations(vertexSource, "attribute");
    const geometry = createParticleGeometry(buildParticleAttributes({ ...BLUEGOLD_CONFIG, count: 4 }), 1);
    const sizeOf: Record<string, number> = { float: 1, vec2: 2, vec3: 3, vec4: 4 };
    expect([...attrs.keys()].sort()).toEqual(["aSeed", "aSize", "aTone"]);
    for (const [name, type] of attrs) {
      expect(geometry.getAttribute(name).itemSize, name).toBe(sizeOf[type]);
    }
    // `position` is declared by three's shader prefix, not by us.
    expect(geometry.getAttribute("position").itemSize).toBe(3);
  });

  it("varyings agree between the two stages", () => {
    expect(declarations(fragmentSource, "varying")).toEqual(declarations(vertexSource, "varying"));
  });

  it("the compiled vertex shader includes the noise function it calls", () => {
    expect(vertexSource).toContain("snoise(");
    expect(VERTEX_SHADER).toMatch(/float snoise\(vec3 v\)/);
    expect(VERTEX_SHADER.indexOf("float snoise(")).toBeLessThan(VERTEX_SHADER.indexOf("void main()"));
  });

  it("the fragment shader converts to the output colour space", () => {
    expect(FRAGMENT_SHADER).toContain("#include <colorspace_fragment>");
  });

  it("animation is driven by uniforms in the shader, not by rewriting buffers", () => {
    expect(vertexSource).toMatch(/uTime/);
    expect(vertexSource).toMatch(/snoise\(/);
    expect(vertexSource).toMatch(/uPointer/);
  });
});
