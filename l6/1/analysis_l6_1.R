set.seed(123)

# Install/load required packages
required_packages <- c("ISLR", "leaps")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(ISLR)
library(leaps)

output_dir <- "."
text_output_path <- file.path(output_dir, "analysis_l6_1_output.txt")

predict.regsubsets <- function(object, newdata, id, ...) {
  model_formula <- as.formula(object$call[[2]])
  mat <- model.matrix(model_formula, newdata)
  coefs <- coef(object, id = id)
  mat[, names(coefs)] %*% coefs
}

prediction_error <- function(i, model, data_obj, subset_idx) {
  pred <- predict(model, data_obj[subset_idx, ], id = i)
  mean((data_obj$Salary[subset_idx] - pred)^2)
}

sink(text_output_path)

cat("=== Selekcja cech dla modeli liniowych (Hitters) ===\n\n")

Hitters <- na.omit(Hitters)
cat("Liczba obserwacji po usunieciu NA:", nrow(Hitters), "\n")
cat("Liczba predyktorow dostepnych do selekcji:", ncol(Hitters) - 1, "\n\n")

# Best subset selection
Hitters_bs <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
Hitters_bs_sum <- summary(Hitters_bs)

bic_min <- which.min(Hitters_bs_sum$bic)
cp_min <- which.min(Hitters_bs_sum$cp)
adjr2_max <- which.max(Hitters_bs_sum$adjr2)

cat("=== Globalnie najlepsze podzbiory (best subset) ===\n")
cat("BIC min: model z", bic_min, "zmiennymi; BIC =", Hitters_bs_sum$bic[bic_min], "\n")
cat("Cp min: model z", cp_min, "zmiennymi; Cp  =", Hitters_bs_sum$cp[cp_min], "\n")
cat("AdjR2 max: model z", adjr2_max, "zmiennymi; AdjR2 =", Hitters_bs_sum$adjr2[adjr2_max], "\n\n")

cat("Wspolczynniki modelu optymalnego wg BIC:\n")
print(coef(Hitters_bs, id = bic_min))
cat("\nWspolczynniki modelu optymalnego wg Cp:\n")
print(coef(Hitters_bs, id = cp_min))
cat("\nWspolczynniki modelu optymalnego wg AdjR2:\n")
print(coef(Hitters_bs, id = adjr2_max))
cat("\n")

# Stepwise forward/backward
Hitters_fwd <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "forward")
Hitters_back <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "backward")

Hitters_fwd_sum <- summary(Hitters_fwd)
Hitters_back_sum <- summary(Hitters_back)

fwd_bic <- which.min(Hitters_fwd_sum$bic)
fwd_cp <- which.min(Hitters_fwd_sum$cp)
fwd_adjr2 <- which.max(Hitters_fwd_sum$adjr2)

back_bic <- which.min(Hitters_back_sum$bic)
back_cp <- which.min(Hitters_back_sum$cp)
back_adjr2 <- which.max(Hitters_back_sum$adjr2)

cat("=== Selekcja krokowa: forward ===\n")
cat("BIC min: ", fwd_bic, "zmiennych; BIC =", Hitters_fwd_sum$bic[fwd_bic], "\n")
cat("Cp min:  ", fwd_cp, "zmiennych; Cp  =", Hitters_fwd_sum$cp[fwd_cp], "\n")
cat("AdjR2 max:", fwd_adjr2, "zmiennych; AdjR2 =", Hitters_fwd_sum$adjr2[fwd_adjr2], "\n\n")

cat("=== Selekcja krokowa: backward ===\n")
cat("BIC min: ", back_bic, "zmiennych; BIC =", Hitters_back_sum$bic[back_bic], "\n")
cat("Cp min:  ", back_cp, "zmiennych; Cp  =", Hitters_back_sum$cp[back_cp], "\n")
cat("AdjR2 max:", back_adjr2, "zmiennych; AdjR2 =", Hitters_back_sum$adjr2[back_adjr2], "\n\n")

cat("Czy stepwise znajduje globalnie najlepsze modele (best subset)?\n")
cat("- BIC: forward =", identical(coef(Hitters_bs, bic_min), coef(Hitters_fwd, fwd_bic)),
    ", backward =", identical(coef(Hitters_bs, bic_min), coef(Hitters_back, back_bic)), "\n")
cat("- Cp:  forward =", identical(coef(Hitters_bs, cp_min), coef(Hitters_fwd, fwd_cp)),
    ", backward =", identical(coef(Hitters_bs, cp_min), coef(Hitters_back, back_cp)), "\n")
cat("- AdjR2: forward =", identical(coef(Hitters_bs, adjr2_max), coef(Hitters_fwd, fwd_adjr2)),
    ", backward =", identical(coef(Hitters_bs, adjr2_max), coef(Hitters_back, back_adjr2)), "\n\n")

# Validation set approach
n <- nrow(Hitters)
train <- sample(c(TRUE, FALSE), n, replace = TRUE)
test <- !train

Hitters_bs_v <- regsubsets(Salary ~ ., data = Hitters[train, ], nvmax = 19)
val_errors <- sapply(1:19, prediction_error, model = Hitters_bs_v, data_obj = Hitters, subset_idx = test)
val_opt <- which.min(val_errors)

cat("=== Metoda zbioru walidacyjnego ===\n")
cat("Optymalna liczba zmiennych:", val_opt, "\n")
cat("Minimalny blad walidacyjny MSE:", min(val_errors), "\n\n")

# 10-fold cross-validation
k <- 10
folds <- sample(1:k, n, replace = TRUE)
val_err <- NULL

for (j in 1:k) {
  fit_bs <- regsubsets(Salary ~ ., data = Hitters[folds != j, ], nvmax = 19)
  err <- sapply(1:19, prediction_error, model = fit_bs, data_obj = Hitters, subset_idx = (folds == j))
  val_err <- rbind(val_err, err)
}

cv_errors <- colMeans(val_err)
cv_opt <- which.min(cv_errors)

cat("=== 10-krotna walidacja krzyzowa ===\n")
cat("Optymalna liczba zmiennych:", cv_opt, "\n")
cat("Minimalny blad CV MSE:", min(cv_errors), "\n\n")

cat("=== Podsumowanie liczby zmiennych (best subset) ===\n")
cat("BIC:", bic_min, "| Cp:", cp_min, "| AdjR2:", adjr2_max,
    "| Validation:", val_opt, "| CV:", cv_opt, "\n")

sink()

# Save numeric outputs as CSV for easy inspection
write.csv(
  data.frame(
    model_size = 1:19,
    bic = Hitters_bs_sum$bic,
    cp = Hitters_bs_sum$cp,
    adjr2 = Hitters_bs_sum$adjr2,
    val_mse = val_errors,
    cv_mse = cv_errors
  ),
  file = file.path(output_dir, "analysis_l6_1_metrics.csv"),
  row.names = FALSE
)

# Save plots
png(file.path(output_dir, "analysis_l6_1_bic_plot.png"), width = 900, height = 600)
plot(Hitters_bs_sum$bic, xlab = "Liczba zmiennych", ylab = "BIC", col = "green", type = "b", pch = 20)
points(bic_min, Hitters_bs_sum$bic[bic_min], col = "red", pch = 9, cex = 1.4)
dev.off()

png(file.path(output_dir, "analysis_l6_1_regsubsets_bic.png"), width = 900, height = 600)
plot(Hitters_bs, scale = "bic")
dev.off()

png(file.path(output_dir, "analysis_l6_1_error_curves.png"), width = 900, height = 600)
plot(1:19, val_errors, type = "b", pch = 19, col = "steelblue",
     xlab = "Liczba zmiennych", ylab = "MSE",
     main = "Validation MSE i CV MSE")
lines(1:19, cv_errors, type = "b", pch = 17, col = "darkorange")
legend("topright", legend = c("Validation", "10-fold CV"),
       col = c("steelblue", "darkorange"), pch = c(19, 17), lty = 1)
dev.off()

cat("Gotowe. Pliki wyjsciowe:\n")
cat("- analysis_l6_1_output.txt\n")
cat("- analysis_l6_1_metrics.csv\n")
cat("- analysis_l6_1_bic_plot.png\n")
cat("- analysis_l6_1_regsubsets_bic.png\n")
cat("- analysis_l6_1_error_curves.png\n")
