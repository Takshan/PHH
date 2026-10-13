#!/usr/bin/env Rscript

file_arg <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
script_dir <- if (length(file_arg)) dirname(normalizePath(file_arg[1])) else getwd()
.libPaths(c(file.path(script_dir, "..", "resources", "Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(kohonen))

args <- commandArgs(trailingOnly = TRUE)
matrix_csv <- args[1]
x_dim <- as.integer(args[2])
y_dim <- as.integer(args[3])
rlen <- as.integer(args[4])
seed <- as.integer(args[5])
out_prefix <- args[6]

X <- as.matrix(read.csv(matrix_csv, row.names = 1, check.names = FALSE))
set.seed(seed)
fit <- som(X, grid = somgrid(x_dim, y_dim, "hexagonal"), rlen = rlen,
           alpha = c(0.25, 0.01), keep.data = TRUE)

qe <- sqrt(mean(fit$distances) / (nrow(X) * ncol(X)))

write.csv(data.frame(gene = rownames(X), unit = fit$unit.classif),
          paste0(out_prefix, "_assign.csv"), row.names = FALSE, quote = FALSE)
codes <- if (is.list(fit$codes)) fit$codes[[1]] else fit$codes
write.csv(data.frame(unit = seq_len(nrow(codes)), codes, check.names = FALSE),
          paste0(out_prefix, "_codebook.csv"), row.names = FALSE, quote = FALSE)
write.csv(data.frame(unit = seq_len(nrow(fit$grid$pts)), fit$grid$pts),
          paste0(out_prefix, "_gridpts.csv"), row.names = FALSE, quote = FALSE)
write.csv(data.frame(iter = seq_len(nrow(fit$changes)), qe = fit$changes[, 1]),
          paste0(out_prefix, "_changes.csv"), row.names = FALSE, quote = FALSE)

cat("kohonen", as.character(packageVersion("kohonen")),
    "| converged QE:", round(qe, 5), "\n")
