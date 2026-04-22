#' Run_novae
#' 
#' This function runs novae.
#' 
#' @export
Run_novae <- function(adata_path = NULL, accelerator = "cuda") {
  proc <- basilisk::basiliskStart(.novae)
  on.exit(basilisk::basiliskStop(proc))
  basilisk::basiliskRun(proc, function(adata_path, accelerator) {
    
    # libraries
    os    <- reticulate::import("os")
    os$environ[["HF_HOME"]]      <- ".cache/huggingface"
    os$environ[["MPLCONFIGDIR"]] <- ".cache/matplotlib"
    novae <- reticulate::import("novae")
    ad    <- reticulate::import("anndata")
    np <- reticulate::import("numpy")
    sp <- reticulate::import("scipy")
    
    # read anndata
    adata <- ad$read_h5ad(adata_path)
    
    # sparse matrix
    adata$X <- sp$sparse$csr_matrix(adata$layers$counts)

    # spatial neighbors
    novae$spatial_neighbors(adata)
    
    # novae
    model <- novae$Novae$from_pretrained("MICS-Lab/novae-human-0")
    n_valid_cells <-  as.integer(adata$n_obs())
    model$swav_head$num_prototypes <- min(
      model$swav_head$num_prototypes,
      n_valid_cells %/% 2L
    )

    model$compute_representations(adata, zero_shot = TRUE, accelerator = accelerator, num_workers = 4L)
    model$assign_domains(adata)
    reticulate::py_to_r(adata)
  }, adata_path = adata_path, accelerator = accelerator)
}

#' 
#' Run_nimbus # based on Nimbus-Inference example notebook https://github.com/angelolab/Nimbus-Inference/blob/main/templates/1_Nimbus_Predict.ipynb 
#' 
#' @export

Run_nimbus <- function() {
  proc <- basilisk::basiliskStart(.nimbus)
  on.exit(basilisk::basiliskStop(proc))
  basilisk::basiliskRun(proc, function() {

  # import libraries
    warnings <- import("warnings")
    warnings$simplefilter("ignore")
    os <- import("os")
    nimbus_mod <- import("nimbus_inference.nimbus")
    Nimbus <- nimbus_mod$Nimbus
    prep_naming_convention <- nimbus_mod$prep_naming_convention
    utils_mod <- import("nimbus_inference.utils")
    MultiplexDataset <- utils_mod$MultiplexDataset
    io_utils <- import("alpineer.io_utils")
    nimbus_inference <- import("nimbus_inference")
    example_dataset <- nimbus_inference$example_dataset
    viewer_mod <- import("nimbus_inference.viewer_widget")
    NimbusViewer <- viewer_mod$NimbusViewer

    # set up base directory
    base_dir <- os$path$normpath(base_dir)

    # download example dataset
    example_dataset$get_example_dataset(dataset="cluster_pixels", save_dir=base_dir, overwrite_existing=FALSE)

    # set up file paths
    tiff_dir <- os$path$join(base_dir, "image_data")
    deepcell_output_dir <- os$path$join(base_dir, "segmentation", "deepcell_output")
    nimbus_output_dir <- os$path$join(base_dir, "nimbus_output")
    os$makedirs(nimbus_output_dir, exist_ok=TRUE)
    io_utils$validate_paths(list(base_dir, tiff_dir, deepcell_output_dir, nimbus_output_dir))

    # channels to include
    include_channels <- c(
      "CD3", "CD4", "CD8", "CD14", "CD20", "CD31", "CD45", "CD68", "CD163", "CK17", "Collagen1",
      "ECAD", "Fibronectin", "GLUT1", "HLADR", "IDO", "Ki67", "PD1", "SMA", "Vim"
    )

    # get FOV names and paths
    fov_names <- os$listdir(tiff_dir)
    fov_names <- Filter(function(x) !startsWith(x, "."), fov_names)
    fov_paths <- lapply(fov_names, function(f) os$path$join(tiff_dir, f))

    # prepare segmentation naming convention
    segmentation_naming_convention <- prep_naming_convention(deepcell_output_dir)

    # test segmentation naming convention
    if (os$path$exists(segmentation_naming_convention(fov_paths[[1]]))) {
      message("Segmentation data exists for fov 0 and naming convention is correct")
    } else {
      message("Segmentation data does not exist for fov 0 or naming convention is incorrect")
    }

    # create dataset
    dataset <- MultiplexDataset(
      fov_paths = fov_paths,
      suffix = ".tiff",
      include_channels = include_channels,
      segmentation_naming_convention = segmentation_naming_convention,
    output_dir = nimbus_output_dir
    )

    # init Nimbus
    nimbus <- Nimbus(
      dataset = dataset,
      save_predictions = TRUE,
      batch_size = 4,
      test_time_aug = TRUE,
      input_shape = as.integer(c(1024, 1024)),
      device = "auto",
      output_dir = nimbus_output_dir,
      compile_model = FALSE,
      mixed_precision = FALSE
    )

    # check inputs
    nimbus$check_inputs()

    # Prepare normalization dictionary
    dataset$prepare_normalization_dict(
      quantile = 0.999,
      n_subset = 20L,
      clip_values = tuple(0L, 2L),
      multiprocessing = TRUE,
      overwrite = TRUE
    )

    # make predictions
    cell_table <- nimbus$predict_fovs()
    return(cell_table)
    })
  }



