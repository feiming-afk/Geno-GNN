# 定义文件路径和自定义名称
base_path <- "data\\reality\\"
suffix <- ".csv"
custom_name <- "mrna"  # 自定义名称

# 加载数据
file_path <- paste0(base_path, custom_name, suffix)
data <- read.csv(file_path)

selected_rows1 <- data[data$sequence <= 8, ] # 2020.3-2020.11 9
selected_rows2 <- data[8 < data$sequence & data$sequence <= 12, ] # 2020.12-2021.3 4
selected_rows3 <- data[13 < data$sequence & data$sequence <= 16, ] # 2021.5-2021.7 3
selected_rows4 <- data[16 < data$sequence & data$sequence <= 21, ] # 2021.8-2021.12 5
selected_rows5 <- data[21 < data$sequence & data$sequence <= 23, ] # 2022.1-2022.2 2
selected_rows6 <- data[23 < data$sequence , ] # 2022.3-2024.2

jianyan <- function(selected_rows) {
  # 定义一个辅助函数，用于拟合模型并返回估计值和 p-value
  fit_and_return_values <- function(response_var) {
    formula <- as.formula(paste(response_var, "~ sequence"))
    model <- lm(formula, data = selected_rows)
    model_summary <- summary(model)
    coefficients_table <- model_summary$coefficients
    
    estimate <- coefficients_table["sequence", "Estimate"]
    p_value <- coefficients_table["sequence", "Pr(>|t|)"]
    intercept_estimate <- coefficients_table["(Intercept)", "Estimate"]
    intercept_p_value <- coefficients_table["(Intercept)", "Pr(>|t|)"]
    # 提取斜率的估计值
    slope_estimate <- coef(model)["sequence"]
    
    # 计算斜率的 95% 置信区间
    slope_confint <- confint(model, "sequence", level = 0.95)
    intercept_confint <- confint(model, "(Intercept)", level = 0.95)
    
    return(list(slope_estimate=slope_estimate,p_value=p_value,confint_min=slope_confint[1],confint_max=slope_confint[2],
                intercept_estimate = intercept_estimate, intercept_p_value = intercept_p_value,
                intercept_conf_min = intercept_confint[1], intercept_conf_max = intercept_confint[2]))
    #return(list(estimate = estimate, p_value = p_value))
  }
  
  # 对每个响应变量调用辅助函数并收集结果
  response_vars <- c("bds", "wts", "vcs", "ba1s", "ba2s", "ba5s")
  values <- lapply(response_vars, fit_and_return_values)
  
  return(values)
}

# 对每个分段调用 jianyan 函数并收集结果
fenduan <- list(selected_rows1, selected_rows2, selected_rows3, selected_rows4, selected_rows5, selected_rows6)
results <- lapply(fenduan, jianyan)

# 将结果转换为两个 DataFrame
p_values_df <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "p_value")))
slope_df <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "slope_estimate")))
conf_min_df <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "confint_min")))
conf_max_df <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "confint_max")))
intercept_estimates_all <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "intercept_estimate")))
intercept_p_values_all <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "intercept_p_value")))
intercept_conf_min_all <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "intercept_conf_min")))
intercept_conf_max_all <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "intercept_conf_max")))

rownames(intercept_conf_max_all) <- rownames(intercept_conf_min_all) <- rownames(intercept_estimates_all) <- rownames(intercept_p_values_all) <- rownames(conf_min_df) <- rownames(conf_max_df) <- rownames(slope_df) <- rownames(p_values_df) <- c("Others", "Others > Alpha", "Alpha > Delta", "Delta", "Delta > Omicron", "Omicron")
colnames(intercept_conf_max_all) <- colnames(intercept_conf_min_all) <- colnames(intercept_estimates_all) <- colnames(intercept_p_values_all) <-colnames(conf_min_df) <- colnames(conf_max_df) <- colnames(slope_df) <- colnames(p_values_df) <- c("bds", "wts", "vcs", "ba1s", "ba2s", "ba5s")

# 保存结果到 CSV 文件
output_dir <- paste0("results\\reality\\", custom_name, "\\")
paste0(output_dir, "intercept_estimates_all", suffix)
# 保存估计值 DataFrame 到 CSV 文件
write.csv(p_values_df, paste0(output_dir, "p_values_df", suffix), row.names = TRUE)
write.csv(slope_df, paste0(output_dir, "slope_df", suffix), row.names = TRUE)
write.csv(conf_min_df, paste0(output_dir, "conf_min_df", suffix), row.names = TRUE)
write.csv(conf_max_df, paste0(output_dir, "conf_max_df", suffix), row.names = TRUE)
write.csv(intercept_estimates_all, paste0(output_dir, "intercept_estimates_all", suffix), row.names = TRUE)
write.csv(intercept_p_values_all, paste0(output_dir, "intercept_p_values_df", suffix), row.names = TRUE)
write.csv(intercept_conf_min_all, paste0(output_dir, "intercept_conf_min_df", suffix), row.names = TRUE)
write.csv(intercept_conf_max_all, paste0(output_dir, "intercept_conf_max_df", suffix), row.names = TRUE)