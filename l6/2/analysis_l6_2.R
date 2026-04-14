set.seed(123)

# Install/load required packages
required_packages <- c("ISLR", "glmnet")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(ISLR)
library(glmnet)

output_dir <- "."
text_output_path <- file.path(output_dir, "analysis_l6_2_output.txt")

Hitters <- na.omit(Hitters)
X <- model.matrix(Salary ~ ., data = Hitters)[, -1]
y <- Hitters$Salary

sink(text_output_path)

cat("=== Regularyzacja w modelach liniowych (Hitters) ===\n\n")
cat("Liczba obserwacji po usunieciu NA:", nrow(Hitters), "\n")
cat("Liczba predyktorow:", ncol(X), "\n\n")

# Ridge regression on fixed lambda grid
lambda_grid <- 10^seq(10, -2, length.out = 100)
fit_ridge_grid <- glmnet(X, y, alpha = 0, lambda = lambda_grid)

cat("=== Regresja grzbietowa: siatka lambda ===\n")
cat("Wymiar macierzy wspolczynnikow:", paste(dim(coef(fit_ridge_grid)), collapse = " x "), "\n\n")

lambda_50 <- fit_ridge_grid$lambda[50]
coef_ridge_50 <- coef(fit_ridge_grid)[, 50]
norm_50 <- sqrt(sum(coef_ridge_50[-1]^2))

lambda_70 <- fit_ridge_grid$lambda[70]
coef_ridge_70 <- coef(fit_ridge_grid)[, 70]
norm_70 <- sqrt(sum(coef_ridge_70[-1]^2))

cat("Lambda[50] =", lambda_50, "\n")
cat("Norma Euklidesowa wspolczynnikow (bez interceptu) dla lambda[50]:", norm_50, "\n\n")
cat("Lambda[70] =", lambda_70, "\n")
cat("Norma Euklidesowa wspolczynnikow (bez interceptu) dla lambda[70]:", norm_70, "\n\n")

coef_s_50 <- predict(fit_ridge_grid, s = 50, type = "coefficients")
cat("Wspolczynniki ridge dla nowej wartosci lambda s = 50 (pierwsze 20):\n")
print(coef_s_50[1:20, , drop = FALSE])
cat("\n")

# Train/test split
set.seed(1)
n <- nrow(X)
train <- sample(n, n / 2)
test <- setdiff(seq_len(n), train)

fit_ridge <- glmnet(
  X[train, ], y[train], alpha = 0, lambda = lambda_grid, thresh = 1e-12
)

pred_ridge_4 <- predict(fit_ridge, s = 4, newx = X[test, ])
mse_ridge_4 <- mean((pred_ridge_4 - y[test])^2)

pred_null <- mean(y[train])
mse_null <- mean((pred_null - y[test])^2)

pred_ridge_big <- predict(fit_ridge, s = 1e10, newx = X[test, ])
mse_ridge_big <- mean((pred_ridge_big - y[test])^2)

pred_ridge_0 <- predict(
  fit_ridge, x = X[train, ], y = y[train], s = 0, newx = X[test, ], exact = TRUE
)
mse_ridge_0 <- mean((pred_ridge_0 - y[test])^2)

cat("=== Porownanie MSE (test) ===\n")
cat("Ridge (lambda = 4):", mse_ridge_4, "\n")
cat("Model zerowy (srednia treningu):", mse_null, "\n")
cat("Ridge (lambda = 1e10):", mse_ridge_big, "\n")
cat("Ridge/OLS (lambda = 0):", mse_ridge_0, "\n\n")

lm_fit <- lm(y ~ X, subset = train)
cat("Podsumowanie LM (OLS) dla porownania:\n")
print(summary(lm_fit))
cat("\nWspolczynniki ridge dla lambda = 0 (pierwsze 20):\n")
ridge_coef_0 <- predict(
  fit_ridge, x = X[train, ], y = y[train], s = 0, exact = TRUE, type = "coefficients"
)
print(ridge_coef_0[1:20, , drop = FALSE])
cat("\n")

# CV for ridge
set.seed(1)
cv_ridge <- cv.glmnet(X[train, ], y[train], alpha = 0)
lambda_ridge_opt <- cv_ridge$lambda.min
pred_ridge_opt <- predict(fit_ridge, s = lambda_ridge_opt, newx = X[test, ])
mse_ridge_opt <- mean((pred_ridge_opt - y[test])^2)

fit_ridge_full <- glmnet(X, y, alpha = 0)
coef_ridge_opt <- predict(fit_ridge_full, s = lambda_ridge_opt, type = "coefficients")

cat("=== Ridge z CV ===\n")
cat("Optymalne lambda (cv.glmnet):", lambda_ridge_opt, "\n")
cat("MSE test dla lambda.min:", mse_ridge_opt, "\n")
cat("Wspolczynniki ridge dla lambda.min (pierwsze 20):\n")
print(coef_ridge_opt[1:20, , drop = FALSE])
cat("\n")

# Lasso
fit_lasso <- glmnet(X[train, ], y[train], alpha = 1)
cv_lasso <- cv.glmnet(X[train, ], y[train], alpha = 1)
lambda_lasso_opt <- cv_lasso$lambda.min
pred_lasso <- predict(fit_lasso, s = lambda_lasso_opt, newx = X[test, ])
mse_lasso_opt <- mean((pred_lasso - y[test])^2)

fit_lasso_full <- glmnet(X, y, alpha = 1)
coef_lasso_opt <- predict(fit_lasso_full, s = lambda_lasso_opt, type = "coefficients")

cat("=== Lasso z CV ===\n")
cat("Optymalne lambda (cv.glmnet):", lambda_lasso_opt, "\n")
cat("MSE test dla lambda.min:", mse_lasso_opt, "\n")
cat("Wspolczynniki lasso dla lambda.min (pierwsze 20):\n")
print(coef_lasso_opt[1:20, , drop = FALSE])
cat("\n")

# Comparison requested in tasks
comparison <- data.frame(
  model = c(
    "Null model",
    "Ridge lambda=1e10",
    "Ridge lambda=4",
    "Ridge lambda=0 (OLS)",
    "Ridge lambda=cv lambda.min",
    "Lasso lambda=cv lambda.min"
  ),
  test_mse = c(
    mse_null,
    mse_ridge_big,
    mse_ridge_4,
    mse_ridge_0,
    mse_ridge_opt,
    mse_lasso_opt
  )
)

cat("=== Tabela porownawcza MSE (test) ===\n")
print(comparison)
cat("\nNajlepszy model wg test MSE:\n")
print(comparison[which.min(comparison$test_mse), , drop = FALSE])
cat("\n")

cat("Liczba niezerowych wspolczynnikow (bez interceptu):\n")
cat("Ridge lambda.min:", sum(abs(as.numeric(coef_ridge_opt[-1, 1])) > 0), "\n")
cat("Lasso lambda.min:", sum(abs(as.numeric(coef_lasso_opt[-1, 1])) > 0), "\n")

sink()

# Save tabular outputs
write.csv(comparison, file = file.path(output_dir, "analysis_l6_2_mse_comparison.csv"), row.names = FALSE)

coef_compare <- data.frame(
  variable = rownames(coef_ridge_opt),
  ridge_lambda_min = as.numeric(coef_ridge_opt[, 1]),
  lasso_lambda_min = as.numeric(coef_lasso_opt[, 1])
)
write.csv(coef_compare, file = file.path(output_dir, "analysis_l6_2_coef_comparison.csv"), row.names = FALSE)

# Save plots
png(file.path(output_dir, "analysis_l6_2_ridge_paths.png"), width = 900, height = 600)
plot(fit_ridge_grid, xvar = "lambda", label = FALSE, main = "Ridge paths")
dev.off()

png(file.path(output_dir, "analysis_l6_2_ridge_cv.png"), width = 900, height = 600)
plot(cv_ridge, main = "Ridge CV")
dev.off()

png(file.path(output_dir, "analysis_l6_2_lasso_paths.png"), width = 900, height = 600)
plot(fit_lasso, xvar = "lambda", label = FALSE, main = "Lasso paths")
dev.off()

png(file.path(output_dir, "analysis_l6_2_lasso_cv.png"), width = 900, height = 600)
plot(cv_lasso, main = "Lasso CV")
dev.off()

png(file.path(output_dir, "analysis_l6_2_mse_barplot.png"), width = 1000, height = 600)
barplot(
  height = comparison$test_mse,
  names.arg = comparison$model,
  las = 2,
  col = "steelblue",
  ylab = "Test MSE",
  main = "Porownanie test MSE"
)
dev.off()

cat("Gotowe. Pliki wyjsciowe:\n")
cat("- analysis_l6_2_output.txt\n")
cat("- analysis_l6_2_mse_comparison.csv\n")
cat("- analysis_l6_2_coef_comparison.csv\n")
cat("- analysis_l6_2_ridge_paths.png\n")
cat("- analysis_l6_2_ridge_cv.png\n")
cat("- analysis_l6_2_lasso_paths.png\n")
cat("- analysis_l6_2_lasso_cv.png\n")
cat("- analysis_l6_2_mse_barplot.png\n")
