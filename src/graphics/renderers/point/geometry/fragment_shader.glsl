#version 450

const int TEXTURE_COUNT = 30;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texture_coordinates;
layout(location = 2) flat in int texture_index;

layout (set = 1, binding = 0) uniform sampler2D textures[TEXTURE_COUNT];

layout(push_constant) uniform Constants {
    mat4 world;
    vec3 light_position;
} constants;

void main() {

    vec4 diffuse_color;

    for (int index = 0; index < TEXTURE_COUNT; ++index)
        if (texture_index == index)
            diffuse_color = texture(textures[index], texture_coordinates);

    if (diffuse_color.a != 1.0) {
        discard;
    }

    float light_distance = length(position - constants.light_position);
    gl_FragDepth = light_distance / 256;
}
