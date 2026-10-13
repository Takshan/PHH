#!/usr/bin/env Rscript

file_arg <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
script_dir <- if (length(file_arg)) dirname(normalizePath(file_arg[1])) else getwd()
.libPaths(c(file.path(script_dir, "..", "resources", "Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(kohonen))

args <- commandArgs(trailingOnly = TRUE)
matrix_csv <- args[1]
rlen <- as.integer(args[2])
seed <- as.integer(args[3])
out_prefix <- args[4]

X <- as.matrix(read.csv(matrix_csv, row.names = 1, check.names = FALSE))

grids <- list(
  c(3, 3), c(4, 3), c(4, 4), c(5, 4), c(5, 5),
  c(6, 5), c(7, 6), c(8, 7), c(9, 8), c(10, 9),
  c(11, 10), c(12, 11), c(13, 12), c(14, 13), c(15, 14),
  c(16, 15), c(17, 16), c(18, 17), c(19, 18)
)
if (length(args) >= 5 && nzchar(args[5])) {
  gd <- read.csv(args[5])
  grids <- lapply(seq_len(nrow(gd)), function(i) c(gd$x_dim[i], gd$y_dim[i]))
}

rows <- list()
for (g in grids) {
  x_dim <- g[1]
  y_dim <- g[2]
  k <- x_dim * y_dim
  set.seed(seed)
  t0 <- Sys.time()
  fit <- som(X, grid = somgrid(x_dim, y_dim, "hexagonal"), rlen = rlen,
             alpha = c(0.25, 0.01), keep.data = TRUE)
  dt <- as.numeric(Sys.time() - t0, units = "secs")
  qe <- sqrt(mean(fit$distances) / (nrow(X) * ncol(X)))
  rows[[length(rows) + 1]] <- data.frame(
    x_dim = x_dim,
    y_dim = y_dim,
    n_neurons = k,
    samples_per_neuron = nrow(X) / k,
    qe = qe,
    seconds = dt,
    stringsAsFactors = FALSE
  )
  cat(sprintf("%dx%d=%d  QE=%.5f  (%d genes/neuron, %.1fs)\n",
              x_dim, y_dim, k, qe, round(nrow(X) / k), dt))
}

sweep <- do.call(rbind, rows)
write.csv(sweep, paste0(out_prefix, "_sweep.csv"), row.names = FALSE, quote = FALSE)
cat("wrote", paste0(out_prefix, "_sweep.csv"), "\n")
