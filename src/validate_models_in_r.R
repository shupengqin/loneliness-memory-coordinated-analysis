options(contrasts = c("contr.treatment", "contr.poly"))

suppressPackageStartupMessages(library(lme4))

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
root <- if (length(script_arg)) {
  dirname(dirname(normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")))
} else {
  normalizePath(".", winslash = "/")
}
derived_dir <- file.path(root, "outputs", "derived")
out_dir <- file.path(root, "outputs", "r_validation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cohorts <- c("CHARLS", "ELSA", "HRS", "MHAS", "SHARE")

prepare_frame <- function(cohort) {
  path <- file.path(derived_dir, paste0(tolower(cohort), "_long.csv.gz"))
  frame <- read.csv(gzfile(path), stringsAsFactors = FALSE)
  frame$time10 <- frame$time / 10
  baseline <- frame[!duplicated(frame$pid), , drop = FALSE]
  frame$age10 <- (frame$age - mean(baseline$age, na.rm = TRUE)) / 10
  depression_mean <- mean(baseline$depression_excl_lonely, na.rm = TRUE)
  depression_sd <- sd(baseline$depression_excl_lonely, na.rm = TRUE)
  frame$depression_z <-
    (frame$depression_excl_lonely - depression_mean) / depression_sd
  frame$education <- factor(frame$education)
  frame$pid <- factor(frame$pid)
  if (cohort == "SHARE") {
    frame$country_code <- factor(frame$country_code)
  }
  frame
}

build_formula <- function(cohort, model_name) {
  terms <- c("lonely", "age10", "female", "factor(education)", "partnered")
  if (model_name == "full") {
    terms <- c(terms, "depression_z", "diabetes", "stroke")
  }
  if (cohort == "SHARE") {
    terms <- c(terms, "factor(country_code)")
  }
  as.formula(
    paste0(
      "memory_z ~ time10 * (",
      paste(terms, collapse = " + "),
      ") + (1 + time10 | pid)"
    )
  )
}

fit_model <- function(frame, cohort, model_name) {
  formula <- build_formula(cohort, model_name)
  required <- c(
    "memory_z", "time10", "lonely", "age10", "female", "education",
    "partnered", "pid"
  )
  if (model_name == "full") {
    required <- c(required, "depression_z", "diabetes", "stroke")
  }
  if (cohort == "SHARE") {
    required <- c(required, "country_code")
  }
  model_frame <- frame[complete.cases(frame[, required, drop = FALSE]), , drop = FALSE]
  counts <- table(model_frame$pid)
  eligible <- names(counts[counts >= 2])
  model_frame <- droplevels(model_frame[model_frame$pid %in% eligible, , drop = FALSE])

  fit <- lmer(
    formula,
    data = model_frame,
    REML = TRUE,
    control = lmerControl(
      optimizer = "nloptwrap",
      calc.derivs = TRUE,
      optCtrl = list(maxeval = 200000)
    )
  )
  coefficients <- coef(summary(fit))
  target_name <- if ("time10:lonely" %in% rownames(coefficients)) {
    "time10:lonely"
  } else {
    "lonely:time10"
  }
  estimate <- unname(coefficients[target_name, "Estimate"])
  std_error <- unname(coefficients[target_name, "Std. Error"])
  baseline_estimate <- unname(coefficients["lonely", "Estimate"])
  baseline_std_error <- unname(coefficients["lonely", "Std. Error"])
  convergence_messages <- fit@optinfo$conv$lme4$messages
  if (is.null(convergence_messages)) convergence_messages <- character(0)

  data.frame(
    cohort = cohort,
    model = model_name,
    estimate = estimate,
    std_error = std_error,
    ci_low = estimate - qnorm(0.975) * std_error,
    ci_high = estimate + qnorm(0.975) * std_error,
    p_value = 2 * pnorm(abs(estimate / std_error), lower.tail = FALSE),
    baseline_estimate = baseline_estimate,
    baseline_std_error = baseline_std_error,
    participants = length(unique(model_frame$pid)),
    observations = nrow(model_frame),
    converged = length(convergence_messages) == 0,
    singular = isSingular(fit, tol = 1e-4),
    log_likelihood = as.numeric(logLik(fit)),
    residual_variance = sigma(fit)^2,
    convergence_messages = paste(convergence_messages, collapse = " | "),
    stringsAsFactors = FALSE
  )
}

reml_meta <- function(estimates) {
  y <- estimates$estimate
  variance <- estimates$std_error^2
  k <- length(y)
  objective <- function(tau2) {
    weights <- 1 / (variance + tau2)
    mean_estimate <- sum(weights * y) / sum(weights)
    0.5 * (
      sum(log(variance + tau2)) +
        log(sum(weights)) +
        sum(weights * (y - mean_estimate)^2)
    )
  }
  upper <- max(var(y) * 20, max(variance) * 100, 1e-6)
  tau2 <- max(0, optimize(objective, interval = c(0, upper), tol = 1e-12)$minimum)
  weights <- 1 / (variance + tau2)
  pooled <- sum(weights * y) / sum(weights)
  conventional_se <- sqrt(1 / sum(weights))
  q_re <- sum(weights * (y - pooled)^2)
  hk_scale <- q_re / (k - 1)
  hk_se <- sqrt(hk_scale / sum(weights))
  hk_critical <- qt(0.975, df = k - 1)
  fixed_weights <- 1 / variance
  fixed_mean <- sum(fixed_weights * y) / sum(fixed_weights)
  q <- sum(fixed_weights * (y - fixed_mean)^2)
  i2 <- if (q > 0) max(0, (q - (k - 1)) / q) * 100 else 0
  prediction_critical <- qt(0.975, df = max(k - 2, 1))
  prediction_half_width <-
    prediction_critical * sqrt(tau2 + conventional_se^2)
  data.frame(
    k = k,
    pooled_estimate = pooled,
    hk_std_error = hk_se,
    ci_low = pooled - hk_critical * hk_se,
    ci_high = pooled + hk_critical * hk_se,
    p_value = 2 * pt(abs(pooled / hk_se), df = k - 1, lower.tail = FALSE),
    tau2_reml = tau2,
    i2_percent = i2,
    q = q,
    q_df = k - 1,
    q_p_value = pchisq(q, df = k - 1, lower.tail = FALSE),
    prediction_low = pooled - prediction_half_width,
    prediction_high = pooled + prediction_half_width
  )
}

rows <- list()
row_index <- 1
for (cohort in cohorts) {
  frame <- prepare_frame(cohort)
  for (model_name in c("core", "full")) {
    message("Fitting ", cohort, " ", model_name)
    rows[[row_index]] <- fit_model(frame, cohort, model_name)
    row_index <- row_index + 1
  }
}

cohort_results <- do.call(rbind, rows)
write.csv(
  cohort_results,
  file.path(out_dir, "r_cohort_model_estimates.csv"),
  row.names = FALSE
)

meta_rows <- lapply(c("core", "full"), function(model_name) {
  result <- reml_meta(cohort_results[cohort_results$model == model_name, , drop = FALSE])
  cbind(model = model_name, result)
})
meta_results <- do.call(rbind, meta_rows)
write.csv(
  meta_results,
  file.path(out_dir, "r_meta_analysis_results.csv"),
  row.names = FALSE
)

baseline_input <- cohort_results[cohort_results$model == "core", , drop = FALSE]
baseline_input$estimate <- baseline_input$baseline_estimate
baseline_input$std_error <- baseline_input$baseline_std_error
baseline_meta <- cbind(estimand = "baseline_loneliness", reml_meta(baseline_input))
write.csv(
  baseline_meta,
  file.path(out_dir, "r_baseline_meta_result.csv"),
  row.names = FALSE
)

python_cohort <- read.csv(file.path(root, "outputs", "cohort_model_estimates.csv"))
cohort_comparison <- merge(
  cohort_results,
  python_cohort[, c("cohort", "model", "estimate", "std_error", "participants", "observations")],
  by = c("cohort", "model"),
  suffixes = c("_r", "_python")
)
cohort_comparison$estimate_difference <-
  cohort_comparison$estimate_r - cohort_comparison$estimate_python
cohort_comparison$std_error_difference <-
  cohort_comparison$std_error_r - cohort_comparison$std_error_python
write.csv(
  cohort_comparison,
  file.path(out_dir, "r_python_cohort_comparison.csv"),
  row.names = FALSE
)

python_meta <- read.csv(file.path(root, "outputs", "meta_analysis_results.csv"))
meta_comparison <- merge(
  meta_results,
  python_meta[, c(
    "model", "pooled_estimate", "hk_std_error", "ci_low", "ci_high",
    "p_value", "tau2_reml", "i2_percent", "prediction_low", "prediction_high"
  )],
  by = "model",
  suffixes = c("_r", "_python")
)
for (field in c(
  "pooled_estimate", "hk_std_error", "ci_low", "ci_high", "p_value",
  "tau2_reml", "i2_percent", "prediction_low", "prediction_high"
)) {
  meta_comparison[[paste0(field, "_difference")]] <-
    meta_comparison[[paste0(field, "_r")]] -
    meta_comparison[[paste0(field, "_python")]]
}
write.csv(
  meta_comparison,
  file.path(out_dir, "r_python_meta_comparison.csv"),
  row.names = FALSE
)

figure_source <- read.csv(file.path(root, "outputs", "figures", "figure1_source_data.csv"))
python_baseline <- figure_source[
  figure_source$panel == "b" & figure_source$label == "Pooled",
  c("estimate", "ci_low", "ci_high")
]
baseline_comparison <- data.frame(
  estimand = "baseline_loneliness",
  pooled_estimate_r = baseline_meta$pooled_estimate,
  pooled_estimate_python = python_baseline$estimate,
  pooled_estimate_difference = baseline_meta$pooled_estimate - python_baseline$estimate,
  ci_low_r = baseline_meta$ci_low,
  ci_low_python = python_baseline$ci_low,
  ci_high_r = baseline_meta$ci_high,
  ci_high_python = python_baseline$ci_high
)
write.csv(
  baseline_comparison,
  file.path(out_dir, "r_python_baseline_comparison.csv"),
  row.names = FALSE
)

apply_sensitivity_filter <- function(frame, analysis) {
  frame <- frame[order(frame$pid, frame$time), , drop = FALSE]
  if (analysis == "exclude_first_retest") {
    followup_index <- which(frame$time > 0)
    first_retest_index <- followup_index[
      !duplicated(frame$pid[followup_index])
    ]
    frame <- frame[-first_retest_index, , drop = FALSE]
  } else if (analysis == "exclude_followup_within_2y") {
    frame <- frame[frame$time == 0 | frame$time > 2, , drop = FALSE]
  } else {
    stop("Unknown sensitivity analysis: ", analysis)
  }
  frame
}

selected_sensitivities <- c(
  "exclude_first_retest",
  "exclude_followup_within_2y"
)
sensitivity_rows <- list()
sensitivity_index <- 1
for (cohort in cohorts) {
  frame <- prepare_frame(cohort)
  for (analysis in selected_sensitivities) {
    message("Fitting ", cohort, " ", analysis)
    filtered <- apply_sensitivity_filter(frame, analysis)
    result <- fit_model(filtered, cohort, "core")
    result$analysis <- analysis
    result$model <- NULL
    sensitivity_rows[[sensitivity_index]] <- result
    sensitivity_index <- sensitivity_index + 1
  }
}

sensitivity_results <- do.call(rbind, sensitivity_rows)
write.csv(
  sensitivity_results,
  file.path(out_dir, "r_selected_sensitivity_cohort_estimates.csv"),
  row.names = FALSE
)

sensitivity_meta <- do.call(rbind, lapply(selected_sensitivities, function(analysis) {
  result <- reml_meta(
    sensitivity_results[
      sensitivity_results$analysis == analysis,
      ,
      drop = FALSE
    ]
  )
  cbind(analysis = analysis, result)
}))
write.csv(
  sensitivity_meta,
  file.path(out_dir, "r_selected_sensitivity_meta_results.csv"),
  row.names = FALSE
)

python_sensitivity <- read.csv(
  file.path(root, "outputs", "sensitivity_cohort_estimates.csv")
)
python_sensitivity <- python_sensitivity[
  python_sensitivity$analysis %in% selected_sensitivities,
  ,
  drop = FALSE
]
sensitivity_comparison <- merge(
  sensitivity_results,
  python_sensitivity[, c(
    "cohort", "analysis", "estimate", "std_error", "participants",
    "observations"
  )],
  by = c("cohort", "analysis"),
  suffixes = c("_r", "_python")
)
sensitivity_comparison$estimate_difference <-
  sensitivity_comparison$estimate_r - sensitivity_comparison$estimate_python
sensitivity_comparison$std_error_difference <-
  sensitivity_comparison$std_error_r - sensitivity_comparison$std_error_python
write.csv(
  sensitivity_comparison,
  file.path(out_dir, "r_python_selected_sensitivity_cohort_comparison.csv"),
  row.names = FALSE
)

python_sensitivity_meta <- read.csv(
  file.path(root, "outputs", "sensitivity_meta_results.csv")
)
python_sensitivity_meta <- python_sensitivity_meta[
  python_sensitivity_meta$analysis %in% selected_sensitivities,
  ,
  drop = FALSE
]
sensitivity_meta_comparison <- merge(
  sensitivity_meta,
  python_sensitivity_meta[, c(
    "analysis", "pooled_estimate", "hk_std_error", "ci_low", "ci_high",
    "p_value", "tau2_reml", "i2_percent", "prediction_low",
    "prediction_high"
  )],
  by = "analysis",
  suffixes = c("_r", "_python")
)
for (field in c(
  "pooled_estimate", "hk_std_error", "ci_low", "ci_high", "p_value",
  "tau2_reml", "i2_percent", "prediction_low", "prediction_high"
)) {
  sensitivity_meta_comparison[[paste0(field, "_difference")]] <-
    sensitivity_meta_comparison[[paste0(field, "_r")]] -
    sensitivity_meta_comparison[[paste0(field, "_python")]]
}
write.csv(
  sensitivity_meta_comparison,
  file.path(out_dir, "r_python_selected_sensitivity_meta_comparison.csv"),
  row.names = FALSE
)

capture.output(sessionInfo(), file = file.path(out_dir, "r_session_info.txt"))
print(cohort_results)
print(meta_results)
print(baseline_meta)
print(cohort_comparison[, c(
  "cohort", "model", "estimate_r", "estimate_python",
  "estimate_difference", "std_error_difference", "singular"
)])
print(meta_comparison[, c(
  "model", "pooled_estimate_r", "pooled_estimate_python",
  "pooled_estimate_difference", "ci_low_r", "ci_high_r",
  "tau2_reml_r", "i2_percent_r"
)])
print(baseline_comparison)
print(sensitivity_results[, c(
  "cohort", "analysis", "estimate", "std_error", "participants",
  "observations", "converged", "singular"
)])
print(sensitivity_meta)
print(sensitivity_comparison[, c(
  "cohort", "analysis", "estimate_r", "estimate_python",
  "estimate_difference", "std_error_difference"
)])
print(sensitivity_meta_comparison[, c(
  "analysis", "pooled_estimate_r", "pooled_estimate_python",
  "pooled_estimate_difference", "ci_low_r", "ci_high_r",
  "i2_percent_r"
)])
