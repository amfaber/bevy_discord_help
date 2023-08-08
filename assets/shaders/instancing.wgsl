#import bevy_pbr::mesh_vertex_output MeshVertexOutput

@group(2) @binding(0)
var texture: texture_depth_2d;

@group(2) @binding(1)
var my_sampler: sampler;

@fragment
fn fragment(in: MeshVertexOutput) -> @location(0) vec4<f32> {
    // let uv = (in.position.xy + 1.0) / 2.0;
    // let sample = textureSample(texture, my_sampler, uv);
    // return vec4(sample);
    // return vec4(in.position.xy / 10000., 0.0, 1.0);
    return vec4(1.);
}
