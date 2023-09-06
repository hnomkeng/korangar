mod entity;
mod geometry;
mod indicator;

use std::sync::Arc;

use cgmath::{Matrix4, Point3, Vector2, Vector3};
use vulkano::device::{DeviceOwned, Queue};
use vulkano::format::{ClearValue, Format};
use vulkano::image::{ImageUsage, SampleCount};
use vulkano::render_pass::RenderPass;

use self::entity::EntityRenderer;
use self::geometry::GeometryRenderer;
use self::indicator::IndicatorRenderer;
use super::SubpassAttachments;
use crate::graphics::{
    EntityRenderer as EntityRendererTrait, GeometryRenderer as GeometryRendererTrait, IndicatorRenderer as IndicatorRendererTrait, *,
};
use crate::loaders::{GameFileLoader, TextureLoader};
use crate::network::EntityId;

#[derive(PartialEq)]
pub enum PointShadowSubrenderer {
    Geometry(Matrix4<f32>, f32),
    Entity,
    Indicator,
}

impl VariantEq for PointShadowSubrenderer {
    fn variant_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Geometry(..), Self::Geometry(..)) => true,
            (Self::Entity, Self::Entity) => true,
            (Self::Indicator, Self::Indicator) => true,
            _ => false,
        }
    }
}

pub struct PointShadowRenderer {
    memory_allocator: Arc<MemoryAllocator>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    geometry_renderer: GeometryRenderer,
    entity_renderer: EntityRenderer,
    indicator_renderer: IndicatorRenderer,
    walk_indicator: Arc<ImageView>,
    position: Point3<f32>,
}

unsafe impl Send for PointShadowRenderer {}
unsafe impl Sync for PointShadowRenderer {}

impl PointShadowRenderer {
    const fn subpass() -> SubpassAttachments {
        SubpassAttachments { color: 0, depth: 1 }
    }

    pub fn new(
        memory_allocator: Arc<MemoryAllocator>,
        game_file_loader: &mut GameFileLoader,
        texture_loader: &mut TextureLoader,
        queue: Arc<Queue>,
    ) -> Self {
        let device = memory_allocator.device().clone();
        let render_pass = vulkano::single_pass_renderpass!(
            device,
            attachments: {
                depth: {
                    format: Format::D32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                }
            },
            pass: {
                color: [],
                depth_stencil: {depth}
            }
        )
        .unwrap();

        let subpass = render_pass.clone().first_subpass();
        let geometry_renderer = GeometryRenderer::new(memory_allocator.clone(), subpass.clone());
        let entity_renderer = EntityRenderer::new(memory_allocator.clone(), subpass.clone());
        let indicator_renderer = IndicatorRenderer::new(memory_allocator.clone(), subpass);

        let walk_indicator = texture_loader.get("grid.tga", game_file_loader).unwrap();

        Self {
            memory_allocator,
            queue,
            render_pass,
            geometry_renderer,
            entity_renderer,
            indicator_renderer,
            walk_indicator,
            position: Point3::new(0.0, 0.0, 0.0),
        }
    }

    pub fn set_position(&mut self, position: Point3<f32>) {
        self.position = position;
    }

    pub fn create_render_target(&self, size: u32, array_layers: u32) -> <Self as Renderer>::Target {
        <Self as Renderer>::Target::new(
            self.memory_allocator.clone(),
            self.queue.clone(),
            self.render_pass.clone(),
            [size; 2],
            array_layers,
            SampleCount::Sample1,
            ImageUsage::SAMPLED | ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ClearValue::Depth(1.0),
        )
    }
}

pub struct PointShadowFormat {}

impl IntoFormat for PointShadowFormat {
    fn into_format() -> Format {
        Format::D32_SFLOAT
    }
}

impl Renderer for PointShadowRenderer {
    type Target = SingleRenderTarget<PointShadowFormat, PointShadowSubrenderer, ClearValue>;
}

impl GeometryRendererTrait for PointShadowRenderer {
    fn render_geometry(
        &self,
        render_target: &mut <Self as Renderer>::Target,
        camera: &dyn Camera,
        vertex_buffer: Subbuffer<[ModelVertex]>,
        textures: &[Arc<ImageView>],
        world_matrix: Matrix4<f32>,
        time: f32,
    ) where
        Self: Renderer,
    {
        self.geometry_renderer.render(
            render_target,
            camera,
            self.position,
            vertex_buffer,
            textures,
            world_matrix,
            time,
        );
    }
}

impl EntityRendererTrait for PointShadowRenderer {
    fn render_entity(
        &self,
        render_target: &mut <Self as Renderer>::Target,
        camera: &dyn Camera,
        texture: Arc<ImageView>,
        position: Vector3<f32>,
        origin: Vector3<f32>,
        scale: Vector2<f32>,
        cell_count: Vector2<usize>,
        cell_position: Vector2<usize>,
        mirror: bool,
        _entity_id: EntityId,
    ) where
        Self: Renderer,
    {
        self.entity_renderer.render(
            render_target,
            camera,
            self.position,
            texture,
            position,
            origin,
            scale,
            cell_count,
            cell_position,
            mirror,
        );
    }
}

impl IndicatorRendererTrait for PointShadowRenderer {
    fn render_walk_indicator(
        &self,
        render_target: &mut <Self as Renderer>::Target,
        camera: &dyn Camera,
        _color: Color,
        upper_left: Vector3<f32>,
        upper_right: Vector3<f32>,
        lower_left: Vector3<f32>,
        lower_right: Vector3<f32>,
    ) where
        Self: Renderer,
    {
        self.indicator_renderer.render_ground_indicator(
            render_target,
            camera,
            self.position,
            self.walk_indicator.clone(),
            upper_left,
            upper_right,
            lower_left,
            lower_right,
        );
    }
}
