def evaluate_sla(
    response_time: float,
    error_rate: float,
    cpu_usage: float,
    memory_usage: float,
    target_response_time: float = 200.0,
    maximum_error_rate: float = 1.0,
    minimum_availability: float = 99.0
) -> dict:
    """
    Evaluates application performance against SLA (Service Level Agreement) targets.
    
    Returns SLA status (VIOLATED, AT_RISK, HEALTHY) and a standardized risk score [0.0, 1.0].
    """
    # 1. Compute Availability
    # Availability is defined as the percentage of successful requests
    availability = 100.0 - error_rate
    
    # 2. Check SLA Violations (Out of compliance)
    has_rt_violation = response_time >= target_response_time
    has_err_violation = error_rate >= maximum_error_rate
    has_avail_violation = availability < minimum_availability
    
    is_violated = has_rt_violation or has_err_violation or has_avail_violation
    
    # 3. Check SLA Risk Warnings (Approaching thresholds or resource saturation)
    is_rt_at_risk = response_time >= (0.8 * target_response_time)
    is_err_at_risk = error_rate >= (0.8 * maximum_error_rate)
    is_avail_at_risk = availability < (minimum_availability + 0.5)
    is_cpu_saturated = cpu_usage >= 80.0
    is_mem_saturated = memory_usage >= 85.0
    
    is_at_risk = is_rt_at_risk or is_err_at_risk or is_avail_at_risk or is_cpu_saturated or is_mem_saturated
    
    # 4. Determine SLA Status
    if is_violated:
        status = "VIOLATED"
    elif is_at_risk:
        status = "AT_RISK"
    else:
        status = "HEALTHY"
        
    # 5. Calculate continuous Risk Score in range [0.0, 1.0]
    # Ratio values
    rt_ratio = min(1.0, response_time / target_response_time)
    err_ratio = min(1.0, error_rate / maximum_error_rate)
    
    # Deficit ratio for availability
    avail_deficit = max(0.0, minimum_availability - availability)
    avail_max_deficit = 100.0 - minimum_availability
    avail_ratio = min(1.0, avail_deficit / avail_max_deficit) if avail_max_deficit > 0 else 0.0
    
    cpu_ratio = cpu_usage / 100.0
    mem_ratio = memory_usage / 100.0
    
    # Risk is the max exposure across performance indicators
    risk_score = max(rt_ratio, err_ratio, avail_ratio, cpu_ratio, mem_ratio)
    risk_score = max(0.0, min(1.0, risk_score))
    
    # Force risk to 1.0 if SLA is explicitly violated
    if is_violated:
        risk_score = 1.0
        
    # Build detailed reasoning log
    reasons = []
    if has_rt_violation:
        reasons.append(f"Response time violated: {response_time:.1f}ms >= {target_response_time}ms")
    elif is_rt_at_risk:
        reasons.append(f"Response time high: {response_time:.1f}ms (target: {target_response_time}ms)")
        
    if has_err_violation:
        reasons.append(f"Error rate violated: {error_rate:.2f}% >= {maximum_error_rate}%")
    elif is_err_at_risk:
        reasons.append(f"Error rate high: {error_rate:.2f}% (max limit: {maximum_error_rate}%)")
        
    if has_avail_violation:
        reasons.append(f"Availability violated: {availability:.2f}% < {minimum_availability}%")
        
    if is_cpu_saturated:
        reasons.append(f"CPU resource saturated: {cpu_usage:.1f}% >= 80%")
    if is_mem_saturated:
        reasons.append(f"Memory resource saturated: {memory_usage:.1f}% >= 85%")
        
    reason_str = "; ".join(reasons) if reasons else "Application performance is healthy and well within SLA bounds."
    
    return {
        "status": status,
        "risk_score": round(risk_score, 4),
        "availability": round(availability, 4),
        "reason": reason_str
    }
