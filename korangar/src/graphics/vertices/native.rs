use cgmath::{EuclideanSpace, InnerSpace, Point2, Point3, Vector2, Vector3};
use derive_new::new;
use korangar_util::texture_atlas::AtlasAllocation;
use smallvec::{SmallVec, smallvec_inline};

use crate::graphics::{Color, ModelVertex};

#[derive(Clone, new)]
pub struct NativeModelVertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
    pub texture_coordinates: Vector2<f32>,
    pub texture_index: i32,
    pub color: Color,
    pub wind_affinity: f32,
    pub smoothing_groups: SmallVec<[i32; 3]>,
}

impl NativeModelVertex {
    pub const fn zeroed() -> NativeModelVertex {
        NativeModelVertex {
            position: Point3::new(0.0, 0.0, 0.0),
            normal: Vector3::new(0.0, 0.0, 0.0),
            texture_coordinates: Vector2::new(0.0, 0.0),
            texture_index: 0,
            color: Color::rgba(0.0, 0.0, 0.0, 0.0),
            wind_affinity: 0.0,
            smoothing_groups: smallvec_inline![0; 3],
        }
    }

    fn convert_to_vertex(self, texture_mapping: &[AtlasAllocation]) -> ModelVertex {
        let allocation = texture_mapping[self.texture_index as usize];

        ModelVertex::new(
            self.position,
            self.normal,
            allocation.map_to_atlas(Point2::from_vec(self.texture_coordinates)).to_vec(),
            self.color,
            self.wind_affinity,
        )
    }

    pub fn to_vertices(mut native_vertices: Vec<NativeModelVertex>, texture_mapping: &[AtlasAllocation]) -> Vec<ModelVertex> {
        let mut vertices = Vec::new();
        let mut drain_iterator = native_vertices.drain(..);

        while let Some(mut first_partial) = drain_iterator.next() {
            let mut second_partial = drain_iterator.next().unwrap();
            let mut third_partial = drain_iterator.next().unwrap();

            first_partial.normal = first_partial.normal.normalize();
            second_partial.normal = second_partial.normal.normalize();
            third_partial.normal = third_partial.normal.normalize();

            vertices.push(first_partial.convert_to_vertex(texture_mapping));
            vertices.push(second_partial.convert_to_vertex(texture_mapping));
            vertices.push(third_partial.convert_to_vertex(texture_mapping));
        }

        vertices
    }

    pub fn calculate_normal(first_position: Point3<f32>, second_position: Point3<f32>, third_position: Point3<f32>) -> Vector3<f32> {
        let delta_position_1 = second_position - first_position;
        let delta_position_2 = third_position - first_position;
        delta_position_1.cross(delta_position_2)
    }
}
