# 定义文件路径和自定义名称
base_path <- "data\\random\\jianyan_"
suffix <- ".csv"
custom_name <- "aden_random_mrna_random"  # 自定义名称

# 辅助函数：拟合模型并返回截距、dum 和 dumseq 的估计值和 p 值
fit_and_return_values <- function(response_var, selected_rows) {
  formula <- as.formula(paste(response_var, "~ sequence + dum + dumseq"))
  model <- lm(formula, data = selected_rows)
  model_summary <- summary(model)
  coefficients_table <- model_summary$coefficients
  
  # 提取所需统计信息
  intercept_estimate <- coefficients_table["(Intercept)", "Estimate"]
  intercept_p_value <- coefficients_table["(Intercept)", "Pr(>|t|)"]
  dum_estimate <- coefficients_table["dum", "Estimate"]
  dum_p_value <- coefficients_table["dum", "Pr(>|t|)"]
  dumseq_estimate <- coefficients_table["dumseq", "Estimate"]
  dumseq_p_value <- coefficients_table["dumseq", "Pr(>|t|)"]
  
  slope_info <- slope_values(response_var, selected_rows)
  
  return(list(
    intercept_estimate = intercept_estimate, 
    intercept_p_value = intercept_p_value,
    dum_estimate = dum_estimate,
    dum_p_value = dum_p_value,
    dumseq_estimate = dumseq_estimate,
    dumseq_p_value = dumseq_p_value,
    slope_estimate = slope_info$slope_estimate, 
    confint_min = slope_info$confint_min, 
    confint_max = slope_info$confint_max
  ))
}

# 辅助函数：计算斜率和 95% 置信区间
slope_values <- function(response_var, selected_rows) {
  formula <- as.formula(paste(response_var, "~ sequence"))
  model <- lm(formula, data = selected_rows)
  
  # 提取斜率估计值及其置信区间
  slope_estimate <- coef(model)["sequence"]
  slope_confint <- confint(model, "sequence", level = 0.95)
  
  return(list(slope_estimate = slope_estimate, confint_min = slope_confint[1], confint_max = slope_confint[2]))
}

# 主函数：对一个分段数据进行分析并返回结果
jianyan <- function(selected_rows) {
  response_vars <- c("affinity", "wt", "ina")
  values <- lapply(response_vars, function(var) fit_and_return_values(var, selected_rows))
  return(values)
}

# 加载数据
file_path <- paste0(base_path, custom_name, suffix)
data <- read.csv(file_path)

# 初始化结果列表
all_results <- list()

# 遍历每个 random_state 的值，进行分段分析
for (state in 0:99) {
  state_data <- subset(data, random_state == state)
  
  # 根据条件筛选分段数据
  selected_rows1 <- state_data[8 < state_data$sequence & state_data$sequence <= 12, ]
  selected_rows2 <- state_data[13 < state_data$sequence & state_data$sequence <= 16, ]
  
  # 对每个分段调用 jianyan 函数并收集结果
  fenduan <- list(selected_rows1, selected_rows2)
  results <- lapply(fenduan, jianyan)
  
  # 将每个 state 的结果存入 all_results
  all_results[[as.character(state)]] <- results
}

# 合并所有结果
combine_results <- function(all_results, metric) {
  do.call(rbind, lapply(all_results, function(results) do.call(rbind, lapply(results, function(x) sapply(x, "[[", metric)))))
}

# 生成所有合并的数据框
intercept_estimates_all <- combine_results(all_results, "intercept_estimate")
intercept_p_values_all <- combine_results(all_results, "intercept_p_value")
dum_estimates_all <- combine_results(all_results, "dum_estimate")
dum_p_values_all <- combine_results(all_results, "dum_p_value")
dumseq_estimates_all <- combine_results(all_results, "dumseq_estimate")
dumseq_p_values_all <- combine_results(all_results, "dumseq_p_value")

# 保存结果到 CSV 文件
output_dir <- paste0("results\\random\\", custom_name, "\\")

# 定义行名（表示分段和 random_state）和列名（表示响应变量）
# 添加 random_state 和 segment 列
n_segments <- 2
random_states <- rep(0:99, each = n_segments)
segments <- rep(1:n_segments, times = 100)

# 为每个数据框添加 random_state 和 segment 列
colnames(intercept_estimates_all) <- colnames(intercept_p_values_all) <- 
  colnames(dum_estimates_all) <- colnames(dum_p_values_all) <- 
  colnames(dumseq_estimates_all) <- colnames(dumseq_p_values_all) <- 
  c("affinity", "wt", "ina")

intercept_estimates_all <- cbind(random_state = random_states, segment = segments, intercept_estimates_all)
intercept_p_values_all <- cbind(random_state = random_states, segment = segments, intercept_p_values_all)
dum_estimates_all <- cbind(random_state = random_states, segment = segments, dum_estimates_all)
dum_p_values_all <- cbind(random_state = random_states, segment = segments, dum_p_values_all)
dumseq_estimates_all <- cbind(random_state = random_states, segment = segments, dumseq_estimates_all)
dumseq_p_values_all <- cbind(random_state = random_states, segment = segments, dumseq_p_values_all)

write.csv(intercept_estimates_all, paste0(output_dir, "intercept_estimates_all", suffix), row.names = F)
write.csv(dum_p_values_all, paste0(output_dir, "dum_p_values_all", suffix), row.names = F)
write.csv(dumseq_p_values_all, paste0(output_dir, "dumseq_p_values_all", suffix), row.names = F)
