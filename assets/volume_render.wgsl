#import bevy_pbr::mesh_vertex_output MeshVertexOutput

#import bevy_pbr::mesh_functions  mesh_position_local_to_clip
#import bevy_pbr::mesh_bindings   mesh

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

// struct VertexOutput {
//     @builtin(position) clip_position: vec4<f32>,
//     @location(0) color: vec4<f32>,
// };

@vertex
fn vertex(vertex: Vertex) -> MeshVertexOutput {
    let position = vertex.position;
    var out: MeshVertexOutput;
    out.position = mesh_position_local_to_clip(
        mesh.model, 
        vec4<f32>(position, 1.0)
    );
    // out.color = vertex.i_color;
    return out;
}


@fragment
fn fragment(input: MeshVertexOutput) -> @location(0) vec4<f32> {
    return input.position;
}
