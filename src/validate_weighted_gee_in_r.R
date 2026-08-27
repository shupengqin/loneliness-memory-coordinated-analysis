script_arg <- grep("^--file=", commandArgs(), value = TRUE)
root <- if (length(script_arg)) {
  dirname(dirname(normalizePath(sub("^--file=", "", script_arg[[1]]), winslash = "/")))
} else {
  normalizePath(".", winslash = "/")
}
local_library <- file.path(root, "work", "r_libs")
if (dir.exists(local_library)) {
  .libPaths(c(local_library, .libPaths()))
}

suppressPackageStartupMessages(library(geepack))

options(contrasts = c("contr.treatment", "contr.poly"))

derived_dir <- file.path(root, "outputs", "derived")
out_dir <- file.path(root, "outputs", "r_validation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cohorts <- c("CHARLS", "ELSA", "HRS", "MHAS", "SHARE")
analyses <- c("gee_unweighted", "gee_attrition_ipw", "gee_survey_x_attrition_ipw")

load_panel <- function(cohort) {
  path <- file.path(
    derived_dir,
    paste0(tolower(cohort), "_attrition_panel.csv.gz")
  )
  panel <- read.csv(gzfile(path), stringsAsFactors = FALSE)
  if (!is.logical(panel$observed)) {
    panel$observed <- tolower(as.character(panel$observed)) == "true"
  }
  panel <- panel[order(panel$pid, panel$time), , drop = FALSE]
  panel$time10 <- panel$time / 10
  baseline <- panel[!duplicated(panel$pid), , drop = FALSE]
  panel$age10 <- (panel$age - mean(baseline$age, na.rm = TRUE)) / 10
  panel$education <- factor(panel$education)
  panel$pid <- factor(panel$pid)
  if (cohort == "SHARE") {
    panel$country_code <- factor(panel$country_code)
  }
  panel
}

estimate_weights <- function(panel, cohort) {
  result <- panel
  result$ipow <- 1
  followup_index <- which(result$time > 0)
  followup <- result[followup_index, , drop = FALSE]
  followup$observed_int <- as.integer(followup$observed)

  schedule_terms <- if (cohort == "SHARE") {
    "factor(wave) + factor(country_code)"
  } else {
    "factor(wave)"
  }
  denominator_formula <- as.formula(paste0(
    "observed_int ~ ", schedule_terms,
    " + lonely + baseline_memory + age10 + female + factor(education) + partnered"
  ))
  numerator_formula <- as.formula(paste0("observed_int ~ ", schedule_terms))

  denominator_fit <- glm(
    denominator_formula,
    data = followup,
    family = binomial()
  )
  numerator_fit <- glm(
    numerator_formula,
    data = followup,
    family = binomial()
  )
  denominator_probability <- pmin(
    pmax(predict(denominator_fit, type = "response"), 0.01),
    0.99
  )
  numerator_probability <- pmin(
    pmax(predict(numerator_fit, type = "response"), 0.01),
    0.99
  )
  if (
    length(denominator_probability) != nrow(followup) ||
      length(numerator_probability) != nrow(followup)
  ) {
    stop("Observation-probability predictions do not align with follow-up rows")
  }
  result$ipow[followup_index] <-
    numerator_probability / denominator_probability

  observed_followup <- result$observed & result$time > 0
  truncation <- quantile(
    result$ipow[observed_followup],
    probs = c(0.01, 0.99),
    type = 7,
    na.rm = TRUE
  )
  result$ipow_truncated <- pmin(
    pmax(result$ipow, unname(truncation[1])),
    unname(truncation[2])
  )

  baseline <- result[!duplicated(result$pid), , drop = FALSE]
  positive_weights <- baseline$baseline_weight[
    !is.na(baseline$baseline_weight) & baseline$baseline_weight > 0
  ]
  baseline$survey_weight <-
    baseline$baseline_weight / mean(positive_weights)
  survey_map <- setNames(baseline$survey_weight, as.character(baseline$pid))
  result$survey_weight <- survey_map[as.character(result$pid)]
  result$combined_weight <- result$ipow_truncated * result$survey_weight

  diagnostics <- data.frame(
    cohort = cohort,
    denominator_converged = denominator_fit$converged,
    numerator_converged = numerator_fit$converged,
    numerator_min_probability = min(numerator_probability),
    numerator_max_probability = max(numerator_probability),
    ipow_raw_observed_mean = mean(result$ipow[observed_followup]),
    ipow_raw_observed_sd = sd(result$ipow[observed_followup]),
    ipow_raw_observed_min = min(result$ipow[observed_followup]),
    ipow_raw_observed_max = max(result$ipow[observed_followup]),
    ipow_observed_mean = mean(result$ipow_truncated[observed_followup]),
    ipow_observed_sd = sd(result$ipow_truncated[observed_followup]),
    ipow_observed_min = min(result$ipow_truncated[observed_followup]),
    ipow_observed_max = max(result$ipow_truncated[observed_followup]),
    ipow_truncation_low = unname(truncation[1]),
    ipow_truncation_high = unname(truncation[2]),
    denominator_min_probability = min(denominator_probability),
    denominator_max_probability = max(denominator_probability)
  )
  list(panel = result, diagnostics = diagnostics)
}

outcome_formula <- function(cohort) {
  terms <- c("lonely", "age10", "female", "factor(education)", "partnered")
  if (cohort == "SHARE") {
    terms <- c(terms, "factor(country_code)")
  }
  as.formula(paste0(
    "memory_z ~ time10 * (",
    paste(terms, collapse = " + "),
    ")"
  ))
}

fit_gee <- function(panel, cohort, analysis) {
  observed <- panel[panel$observed & !is.na(panel$memory_z), , drop = FALSE]
  counts <- table(observed$pid)
  eligible <- names(counts[counts >= 2])
  observed <- observed[observed$pid %in% eligible, , drop = FALSE]

  weight_column <- switch(
    analysis,
    gee_unweighted = NULL,
    gee_attrition_ipw = "ipow_truncated",
    gee_survey_x_attrition_ipw = "combined_weight"
  )
  if (!is.null(weight_column)) {
    observed <- observed[
      !is.na(observed[[weight_column]]) & observed[[weight_column]] > 0,
      ,
      drop = FALSE
    ]
    model_weights <- observed[[weight_column]]
  } else {
    model_weights <- rep(1, nrow(observed))
  }
  sort_index <- order(observed$pid, observed$time)
  observed <- droplevels(observed[sort_index, , drop = FALSE])
  model_weights <- model_weights[sort_index]
  observed$.model_weight <- model_weights

  fit <- geeglm(
    formula = outcome_formula(cohort),
    id = pid,
    data = observed,
    family = gaussian(),
    corstr = "exchangeable",
    weights = .model_weight,
    std.err = "san.se"
  )
  coefficient_table <- summary(fit)$coefficients
  target <- if ("time10:lonely" %in% rownames(coefficient_table)) {
    "time10:lonely"
  } else {
    "lonely:time10"
  }
  estimate <- unname(coefficient_table[target, "Estimate"])
  std_error <- unname(coefficient_table[target, "Std.err"])
  data.frame(
    cohort = cohort,
    analysis = analysis,
    estimate = estimate,
    std_error = std_error,
    ci_low = estimate - qnorm(0.975) * std_error,
    ci_high = estimate + qnorm(0.975) * std_error,
    p_value = 2 * pnorm(abs(estimate / std_error), lower.tail = FALSE),
    participants = length(unique(observed$pid)),
    observations = nrow(observed),
    converged = fit$geese$error == 0,
    dependence_parameter = unname(fit$geese$alpha[1]),
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
  tau2 <- max(0, optimize(objective, c(0, upper), tol = 1e-12)$minimum)
  weights <- 1 / (variance + tau2)
  pooled <- sum(weights * y) / sum(weights)
  conventional_se <- sqrt(1 / sum(weights))
  q_re <- sum(weights * (y - pooled)^2)
  hk_se <- sqrt((q_re / (k - 1)) / sum(weights))
  hk_critical <- qt(0.975, df = k - 1)
  fixed_weights <- 1 / variance
  fixed_mean <- sum(fixed_weights * y) / sum(fixed_weights)
  q <- sum(fixed_weights * (y - fixed_mean)^2)
  i2 <- if (q > 0) max(0, (q - (k - 1)) / q) * 100 else 0
  prediction_half_width <-
    qt(0.975, df = max(k - 2, 1)) * sqrt(tau2 + conventional_se^2)
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

model_rows <- list()
diagnostic_rows <- list()
row_index <- 1
for (cohort in cohorts) {
  weighted <- estimate_weights(load_panel(cohort), cohort)
  diagnostic_rows[[length(diagnostic_rows) + 1]] <- weighted$diagnostics
  for (analysis in analyses) {
    message("Fitting ", cohort, " ", analysis)
    model_rows[[row_index]] <- fit_gee(weighted$panel, cohort, analysis)
    row_index <- row_index + 1
  }
}

cohort_results <- do.call(rbind, model_rows)
diagnostics <- do.call(rbind, diagnostic_rows)
write.csv(
  cohort_results,
  file.path(out_dir, "r_weighted_gee_cohort_estimates.csv"),
  row.names = FALSE
)
write.csv(
  diagnostics,
  file.path(out_dir, "r_attrition_weight_diagnostics.csv"),
  row.names = FALSE
)

python_diagnostics <- read.csv(
  file.path(root, "outputs", "attrition_weight_diagnostics.csv")
)
diagnostic_fields <- c(
  "ipow_observed_mean", "ipow_observed_sd", "ipow_observed_min",
  "ipow_observed_max", "ipow_truncation_low", "ipow_truncation_high",
  "denominator_min_probability", "denominator_max_probability"
)
diagnostic_comparison <- merge(
  diagnostics[, c("cohort", diagnostic_fields)],
  python_diagnostics[, c("cohort", diagnostic_fields)],
  by = "cohort",
  suffixes = c("_r", "_python")
)
for (field in diagnostic_fields) {
  diagnostic_comparison[[paste0(field, "_difference")]] <-
    diagnostic_comparison[[paste0(field, "_r")]] -
    diagnostic_comparison[[paste0(field, "_python")]]
}
write.csv(
  diagnostic_comparison,
  file.path(out_dir, "r_python_attrition_weight_diagnostic_comparison.csv"),
  row.names = FALSE
)

meta_results <- do.call(rbind, lapply(analyses, function(analysis) {
  result <- reml_meta(cohort_results[cohort_results$analysis == analysis, , drop = FALSE])
  cbind(analysis = analysis, result)
}))
write.csv(
  meta_results,
  file.path(out_dir, "r_weighted_gee_meta_results.csv"),
  row.names = FALSE
)

python_cohort <- read.csv(
  file.path(root, "outputs", "attrition_weighted_cohort_estimates.csv")
)
cohort_comparison <- merge(
  cohort_results,
  python_cohort[, c(
    "cohort", "analysis", "estimate", "std_error", "participants",
    "observations", "dependence_parameter"
  )],
  by = c("cohort", "analysis"),
  suffixes = c("_r", "_python")
)
cohort_comparison$estimate_difference <-
  cohort_comparison$estimate_r - cohort_comparison$estimate_python
cohort_comparison$std_error_difference <-
  cohort_comparison$std_error_r - cohort_comparison$std_error_python
write.csv(
  cohort_comparison,
  file.path(out_dir, "r_python_weighted_gee_cohort_comparison.csv"),
  row.names = FALSE
)

python_meta <- read.csv(
  file.path(root, "outputs", "attrition_weighted_meta_results.csv")
)
meta_comparison <- merge(
  meta_results,
  python_meta[, c(
    "analysis", "pooled_estimate", "hk_std_error", "ci_low", "ci_high",
    "p_value", "tau2_reml", "i2_percent", "prediction_low", "prediction_high"
  )],
  by = "analysis",
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
  file.path(out_dir, "r_python_weighted_gee_meta_comparison.csv"),
  row.names = FALSE
)

capture.output(sessionInfo(), file = file.path(out_dir, "r_gee_session_info.txt"))
print(cohort_results)
print(meta_results)
print(cohort_comparison[, c(
  "cohort", "analysis", "estimate_r", "estimate_python",
  "estimate_difference", "std_error_difference"
)])
print(meta_comparison[, c(
  "analysis", "pooled_estimate_r", "pooled_estimate_python",
  "pooled_estimate_difference", "ci_low_r", "ci_high_r"
)])
