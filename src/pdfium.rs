use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use anyhow::{bail, Context};
use image::DynamicImage;
use pdfium_render::prelude::{
    PdfBitmap, PdfBitmapFormat, PdfPageRenderRotation, PdfRenderConfig, Pdfium,
};

const PDFIUM_ENABLED_ENV: &str = "DOC2MSG_PDFIUM_ENABLED";
const PDFIUM_LIB_PATH_ENV: &str = "DOC2MSG_PDFIUM_LIB_PATH";
const PDFIUM_RENDER_TARGET_WIDTH: i32 = 1800;
const PDFIUM_RENDER_MAX_HEIGHT: i32 = 2400;
const PAGE_DIMENSION_EPSILON: f32 = 0.1;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct PdfiumRuntimeConfig {
    enabled: bool,
    library_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdfiumPageText {
    pub page_number: u32,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct PdfiumRenderedPage {
    pub page_number: u32,
    pub width: u32,
    pub height: u32,
    pub image: DynamicImage,
}

pub fn install_runtime_config(config: &crate::config::RuntimeConfig) {
    if !config.pdfium_enabled && config.pdfium_lib_path.is_none() {
        return;
    }

    let mut guard = write_runtime_override();
    *guard = Some(PdfiumRuntimeConfig {
        enabled: config.pdfium_enabled || config.pdfium_lib_path.is_some(),
        library_path: config
            .pdfium_lib_path
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from),
    });
}

pub fn ensure_available() -> anyhow::Result<()> {
    let _ = bind_pdfium()?;
    Ok(())
}

pub fn available() -> bool {
    ensure_available().is_ok()
}

pub async fn extract_text_from_bytes(bytes: Vec<u8>) -> anyhow::Result<Vec<PdfiumPageText>> {
    tokio::task::spawn_blocking(move || extract_text_from_bytes_blocking(bytes))
        .await
        .context("pdfium text worker task failed")?
}

pub async fn render_pages_for_ocr(bytes: Vec<u8>) -> anyhow::Result<Vec<PdfiumRenderedPage>> {
    render_selected_pages_for_ocr(bytes, Vec::new()).await
}

pub async fn render_selected_pages_for_ocr(
    bytes: Vec<u8>,
    page_numbers: Vec<u32>,
) -> anyhow::Result<Vec<PdfiumRenderedPage>> {
    tokio::task::spawn_blocking(move || render_pages_for_ocr_blocking(bytes, page_numbers))
        .await
        .context("pdfium render worker task failed")?
}

fn runtime_override() -> &'static RwLock<Option<PdfiumRuntimeConfig>> {
    static PDFIUM_RUNTIME_OVERRIDE: OnceLock<RwLock<Option<PdfiumRuntimeConfig>>> = OnceLock::new();

    PDFIUM_RUNTIME_OVERRIDE.get_or_init(|| RwLock::new(None))
}

fn read_runtime_override() -> Option<PdfiumRuntimeConfig> {
    match runtime_override().read() {
        Ok(guard) => guard.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    }
}

fn write_runtime_override() -> std::sync::RwLockWriteGuard<'static, Option<PdfiumRuntimeConfig>> {
    match runtime_override().write() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn resolve_runtime_config() -> PdfiumRuntimeConfig {
    read_runtime_override().unwrap_or_else(PdfiumRuntimeConfig::from_environment)
}

impl PdfiumRuntimeConfig {
    fn from_environment() -> Self {
        let library_path = env::var_os(PDFIUM_LIB_PATH_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);

        Self {
            enabled: env_flag(PDFIUM_ENABLED_ENV) || library_path.is_some(),
            library_path,
        }
    }
}

fn env_flag(key: &str) -> bool {
    env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn bind_pdfium() -> anyhow::Result<Pdfium> {
    let config = resolve_runtime_config();

    if !config.enabled {
        bail!(
            "pdfium support is disabled; set {PDFIUM_ENABLED_ENV}=1 or provide {PDFIUM_LIB_PATH_ENV}"
        );
    }

    let bindings = match config.library_path.as_deref() {
        Some(path) => {
            let binding_path = normalize_library_path(path);
            Pdfium::bind_to_library(&binding_path).with_context(|| {
                format!(
                    "failed to bind pdfium from {}",
                    binding_path.to_string_lossy()
                )
            })?
        }
        None => Pdfium::bind_to_system_library()
            .context("failed to bind pdfium from the system library path")?,
    };

    Ok(Pdfium::new(bindings))
}

fn normalize_library_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        Pdfium::pdfium_platform_library_name_at_path(path)
    } else {
        path.to_path_buf()
    }
}

fn extract_text_from_bytes_blocking(bytes: Vec<u8>) -> anyhow::Result<Vec<PdfiumPageText>> {
    let pdfium = bind_pdfium()?;
    let document = pdfium
        .load_pdf_from_byte_vec(bytes, None)
        .context("failed to load PDF bytes into pdfium")?;

    let mut pages = Vec::new();

    for (index, page) in document.pages().iter().enumerate() {
        let page_number = u32::try_from(index + 1).unwrap_or(u32::MAX);
        let text = page
            .text()
            .with_context(|| format!("failed to read pdfium text for page {page_number}"))?
            .all();

        pages.push(PdfiumPageText { page_number, text });
    }

    Ok(pages)
}

fn render_pages_for_ocr_blocking(
    bytes: Vec<u8>,
    page_numbers: Vec<u32>,
) -> anyhow::Result<Vec<PdfiumRenderedPage>> {
    let pdfium = bind_pdfium()?;
    let document = pdfium
        .load_pdf_from_byte_vec(bytes, None)
        .context("failed to load PDF bytes into pdfium for rendering")?;
    let render_config = pdfium_ocr_render_config();
    let selected_pages =
        (!page_numbers.is_empty()).then(|| page_numbers.into_iter().collect::<HashSet<_>>());

    let mut pages = Vec::new();
    let document_bindings = document.bindings();
    let mut reusable_bitmap: Option<PdfBitmap<'_>> = None;
    let mut reusable_page_size: Option<(f32, f32)> = None;

    for (index, page) in document.pages().iter().enumerate() {
        let page_number = u32::try_from(index + 1).unwrap_or(u32::MAX);
        if selected_pages
            .as_ref()
            .is_some_and(|selected| !selected.contains(&page_number))
        {
            continue;
        }

        let current_page_size = (page.width().value, page.height().value);
        let image = if let Some(bitmap) = reusable_bitmap.as_mut() {
            if same_page_dimensions(reusable_page_size, current_page_size) {
                let reuse_config = pdfium_ocr_render_config().set_fixed_size_to_bitmap(bitmap);
                page.render_into_bitmap_with_config(bitmap, &reuse_config)
                    .with_context(|| {
                        format!("failed to render PDF page {page_number} with pdfium")
                    })?;
                bitmap.as_image()
            } else {
                let rendered_bitmap =
                    page.render_with_config(&render_config).with_context(|| {
                        format!("failed to render PDF page {page_number} with pdfium")
                    })?;
                let image = rendered_bitmap.as_image();
                let format = rendered_bitmap
                    .format()
                    .unwrap_or(PdfBitmapFormat::default());
                reusable_page_size = Some(current_page_size);
                reusable_bitmap = Some(PdfBitmap::empty(
                    rendered_bitmap.width(),
                    rendered_bitmap.height(),
                    format,
                    document_bindings,
                )?);
                image
            }
        } else {
            let rendered_bitmap = page
                .render_with_config(&render_config)
                .with_context(|| format!("failed to render PDF page {page_number} with pdfium"))?;
            let image = rendered_bitmap.as_image();
            let format = rendered_bitmap
                .format()
                .unwrap_or(PdfBitmapFormat::default());
            reusable_page_size = Some(current_page_size);
            reusable_bitmap = Some(PdfBitmap::empty(
                rendered_bitmap.width(),
                rendered_bitmap.height(),
                format,
                document_bindings,
            )?);
            image
        };

        pages.push(PdfiumRenderedPage {
            page_number,
            width: image.width(),
            height: image.height(),
            image,
        });
    }

    Ok(pages)
}

fn same_page_dimensions(previous: Option<(f32, f32)>, current: (f32, f32)) -> bool {
    previous.is_some_and(|(previous_width, previous_height)| {
        (previous_width - current.0).abs() <= PAGE_DIMENSION_EPSILON
            && (previous_height - current.1).abs() <= PAGE_DIMENSION_EPSILON
    })
}

fn pdfium_ocr_render_config() -> PdfRenderConfig {
    PdfRenderConfig::new()
        .set_target_width(PDFIUM_RENDER_TARGET_WIDTH)
        .set_maximum_height(PDFIUM_RENDER_MAX_HEIGHT)
        .rotate_if_landscape(PdfPageRenderRotation::Degrees90, true)
}

#[cfg(test)]
mod tests {
    use super::normalize_library_path;

    #[test]
    fn normalize_library_path_keeps_explicit_file() {
        let path = std::path::Path::new("/tmp/libpdfium.so");

        assert_eq!(normalize_library_path(path), path);
    }
}
