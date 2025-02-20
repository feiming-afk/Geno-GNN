# aden_mrna

# 定义基本的文件路径和要替换的部分
base_path <- "data\\reality\\jianyan_"
suffix <- ".csv"
custom_name <- "inac_mrna"  # 在这里修改为你想要的名称

# 拼接新的文件路径
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
    formula <- as.formula(paste(response_var, "~ sequence + dum + dumseq"))
    model <- lm(formula, data = selected_rows)
    model_summary <- summary(model)
    coefficients_table <- model_summary$coefficients
    
    # 提取截距、dum 和 dumseq 的估计值和 p 值
    intercept_estimate <- coefficients_table["(Intercept)", "Estimate"]
    intercept_p_value <- coefficients_table["(Intercept)", "Pr(>|t|)"]
    dum_estimate <- coefficients_table["dum", "Estimate"]
    dum_p_value <- coefficients_table["dum", "Pr(>|t|)"]
    dumseq_estimate <- coefficients_table["dumseq", "Estimate"]
    dumseq_p_value <- coefficients_table["dumseq", "Pr(>|t|)"]
    
    slope_info = slope_values(response_var)
    
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
  
  slope_values <- function(response_var){
    
    formula <- as.formula(paste(response_var, "~ sequence"))
    model <- lm(formula, data = selected_rows)
    
    # 提取斜率的估计值
    slope_estimate <- coef(model)["sequence"]
    
    # 计算斜率的 95% 置信区间
    slope_confint <- confint(model, "sequence", level = 0.95)
    
    return(list(slope_estimate=slope_estimate,confint_min=slope_confint[1],confint_max=slope_confint[2]))
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
intercept_estimates_df <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "intercept_estimate")))
intercept_p_values_df <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "intercept_p_value")))
dum_estimates_df <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "dum_estimate")))
dum_p_values_df <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "dum_p_value")))
dumseq_estimates_df <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "dumseq_estimate")))
dumseq_p_values_df <- do.call(rbind, lapply(results, function(x) sapply(x, "[[", "dumseq_p_value")))

# 对每个分段的p值进行FDR控制
#fdr_p_values_df <- apply(p_values_df, 2, p.adjust, method = "fdr")

# 为 DataFrame 添加行和列名
rownames(intercept_estimates_df) <- rownames(intercept_p_values_df) <- 
  rownames(dum_estimates_df) <- rownames(dum_p_values_df) <- 
  rownames(dumseq_estimates_df) <- rownames(dumseq_p_values_df) <- c("Others", "Others > Alpha", "Alpha > Delta", "Delta", "Delta > Omicron", "Omicron")

colnames(intercept_estimates_df) <- colnames(intercept_p_values_df) <- 
  colnames(dum_estimates_df) <- colnames(dum_p_values_df) <- 
  colnames(dumseq_estimates_df) <- colnames(dumseq_p_values_df) <- c("bds", "wts", "vcs", "ba1s", "ba2s", "ba5s")

# 保存 p-value DataFrame 到 CSV 文件
write.csv(intercept_estimates_df, paste0("results\\reality\\", custom_name, "\\intercept_estimates_df", suffix), row.names = TRUE)
write.csv(dum_p_values_df, paste0("results\\reality\\", custom_name, "\\dum_p_values_df", suffix), row.names = TRUE)
write.csv(dumseq_p_values_df, paste0("results\\reality\\", custom_name, "\\dumseq_p_values_df", suffix), row.names = TRUE)