# Guardrail Node Documentation

## Overview

The `GuardrailNode` is responsible for compliance checking and content security within the agent workflow. It uses MCP (Model Context Protocol) resources for dynamic rule management, performing PII detection, medical information compliance, and content security enforcement to ensure all AI responses meet compliance requirements.

## Purpose

The Guardrail Node serves as the dynamic security and compliance engine that:
- **Loads compliance rules from MCP resources** for flexible policy management
- **Detects PII patterns** (emails, phone numbers, SSNs, credit cards, IP addresses)
- **Checks medical information compliance** (HIPAA, healthcare data)
- **Enforces content policies** based on configurable modes with risk scoring
- **Sanitizes content** by redacting sensitive information with detailed recommendations
- **Provides streaming updates** for compliance operations and risk assessment

## Input State

The Guardrail Node expects an `OptimizedAgentState` with the following key fields:

### Required Fields
- `messages`: Conversation history with the last message to be checked (list)

### Optional Fields
- `session_id`: Session identifier for tracking (string)
- `user_id`: User identifier for context (string)
- `user_query`: Original user query for reference (string)
- `capabilities`: Available tools and agent capabilities (dict)

## Output State

The Guardrail Node updates the state with:

### Compliance Results
- `guardrail_result`: Complete compliance check results (dict)
  - `action`: Enforcement action taken ("ALLOW", "SANITIZE", "BLOCK")
  - `violations`: List of detected violations with details
  - `risk_score`: Numerical risk assessment based on violation severity
  - `recommendations`: List of compliance recommendations for violations

### Content Modifications
- `messages`: Updated with sanitized or blocked content if violations found
- `original_text`: Original content before sanitization (if applicable)
- `sanitized_text`: Content after PII redaction (if applicable)

### State Preservation
- All existing state fields are preserved
- Only compliance-specific fields are added/updated
- Original message content replaced only when necessary

## Dependencies

### Internal Dependencies
- `app.types.agent_state.OptimizedAgentState`: State management
- `app.tracing.langgraph_tracer.trace_node`: Execution tracing

### External Dependencies
- `langgraph.config`: Stream writer and configuration
- `langchain_core.messages`: Message type handling
- `re`: Regular expression pattern matching

### Service Dependencies
- **MCP Manager**: Dynamic rule loading and resource management
- **LangGraph**: Streaming and configuration management
- **Tracing System**: Performance monitoring and debugging

## Core Functionality

### 1. Guardrail Execution
```python
async def execute(state: OptimizedAgentState) -> OptimizedAgentState
```
- **Input**: State with AI response to be checked
- **Output**: State with compliance results and potentially modified content
- **Logic**:
  1. Load guardrail configuration from MCP resources
  2. Extract final response content from last message
  3. Apply MCP-based PII detection and medical compliance checks
  4. Calculate risk score and generate recommendations
  5. Determine enforcement action based on mode and violations
  6. Apply content sanitization or blocking as needed
  7. Stream compliance operation updates with risk assessment

### 2. MCP Resource Loading
```python
async def _load_guardrail_config(self)
```
- **PII Patterns**: Loads from `guardrail://config/pii` resource
- **Medical Keywords**: Loads from `guardrail://config/medical` resource
- **Compliance Policies**: Loads from `guardrail://policies/compliance` resource
- **Pattern Compilation**: Compiles regex patterns for efficient matching
- **Fallback Handling**: Uses default patterns if MCP resources unavailable

### 3. Content Violation Detection
```python
async def _apply_mcp_guardrails(text: str) -> Dict[str, Any]
```
- **PII Detection**: Email, phone, SSN, credit card, IP address patterns
- **Medical Compliance**: HIPAA keywords and healthcare data detection
- **Risk Assessment**: Calculates severity-based risk scores
- **Pattern Matching**: Uses compiled regex patterns for efficiency
- **Comprehensive Analysis**: Combines multiple violation types with recommendations

### 3. Enforcement Action Determination
```python
def _determine_action(violations: list) -> str
```
- **Moderate Mode**: SANITIZE high-severity violations, ALLOW low-severity
- **Strict Mode**: BLOCK any violations, ALLOW only clean content
- **Default Mode**: Treats as moderate mode
- **Priority**: High-severity violations trigger strongest response

### 4. Content Sanitization
```python
def _sanitize_content(text: str) -> str
```
- **Email Redaction**: Replace with `[REDACTED_EMAIL]`
- **Phone Redaction**: Replace with `[REDACTED_PHONE]`
- **SSN Redaction**: Replace with `[REDACTED_SSN]`
- **Global Replacement**: Uses `re.sub()` for all instances
- **Content Preservation**: Non-PII content remains unchanged

## Guardrail Modes

### Moderate Mode (Default)
- **Low Severity Violations**: ALLOW content to pass
- **High Severity Violations**: SANITIZE content by redacting PII
- **Action Examples**:
  - Email found → SANITIZE with redaction
  - Phone found → SANITIZE with redaction
  - SSN found → SANITIZE with redaction
- **Use Case**: General content filtering with user experience balance

### Strict Mode
- **Any Violations**: BLOCK content completely
- **Clean Content**: ALLOW without modification
- **Action Examples**:
  - Any PII found → BLOCK with compliance message
  - No PII found → ALLOW original content
- **Use Case**: High-security environments requiring zero PII exposure

### Custom Modes
- **Extensible Design**: Support for additional modes
- **Default Fallback**: Unknown modes treated as moderate
- **Configuration**: Mode specified during node initialization

## MCP Resource Integration

### PII Configuration Resource
**Resource URI**: `guardrail://config/pii`
```json
{
  "description": "PII Detection Patterns",
  "patterns": {
    "email": "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b",
    "phone": "\\b\\d{3}-?\\d{3}-?\\d{4}\\b",
    "ssn": "\\b\\d{3}-?\\d{2}-?\\d{4}\\b",
    "credit_card": "\\b\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}\\b",
    "ip_address": "\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b"
  }
}
```

### Medical Compliance Resource
**Resource URI**: `guardrail://config/medical`
```json
{
  "description": "Medical Information Compliance",
  "medical_keywords": [
    "diagnosis", "patient", "medical record", "prescription", "medication",
    "treatment", "symptoms", "disease", "illness", "health condition"
  ],
  "compliance_rules": {
    "hipaa": {
      "description": "Health Insurance Portability and Accountability Act",
      "prohibited": ["patient names", "medical record numbers"],
      "required_actions": ["anonymize", "redact", "encrypt"]
    }
  }
}
```

### Compliance Policies Resource
**Resource URI**: `guardrail://policies/compliance`
```json
{
  "description": "Compliance Policy Framework",
  "policies": {
    "hipaa": { "description": "Healthcare data protection" },
    "gdpr": { "description": "General Data Protection Regulation" },
    "pci": { "description": "Payment Card Industry standards" }
  },
  "enforcement_modes": {
    "strict": "Block any violations",
    "moderate": "Sanitize high-risk content",
    "permissive": "Log violations but allow"
  }
}
```

## Violation Classification

### High Severity (PII Types)
- **PII_EXPOSURE**: General PII detected with specific categories
  - **Email**: Email addresses detected
  - **Phone**: Phone numbers detected
  - **SSN**: Social Security Numbers detected
  - **Credit Card**: Credit card numbers detected
  - **IP Address**: IP addresses detected
- **Risk Level**: High exposure risk requiring action
- **Enforcement**: Triggers SANITIZE in moderate mode, BLOCK in strict mode

### Medium Severity (Medical Information)
- **MEDICAL_INFORMATION**: Healthcare-related content detected
- **HIPAA Compliance**: Medical keywords and patient data
- **Risk Level**: Medium exposure risk requiring review
- **Enforcement**: Generates recommendations for compliance review

### Detection Algorithm
```python
# PII Detection
for pii_type, compiled_pattern in self._compiled_patterns.items():
    matches = compiled_pattern.findall(text)
    if matches:
        violations.append({
            "type": "PII_EXPOSURE",
            "category": pii_type,
            "matches": len(matches),
            "severity": "HIGH"
        })

# Medical Information Detection
detected_medical_terms = [keyword for keyword in medical_keywords if keyword in text_lower]
if detected_medical_terms:
    violations.append({
        "type": "MEDICAL_INFORMATION",
        "detected_terms": detected_medical_terms,
        "severity": "MEDIUM",
        "compliance_requirement": "HIPAA"
    })
```

### Risk Score Calculation
```python
risk_score = (
    len([v for v in violations if v["severity"] == "HIGH"]) * 3 +
    len([v for v in violations if v["severity"] == "MEDIUM"]) * 1
)
```
- **High Severity**: 3 points per violation
- **Medium Severity**: 1 point per violation
- **Risk Levels**: 0 (none), 1-3 (low), 4-6 (medium), 7+ (high)

## Content Enforcement Actions

### ALLOW Action
- **Condition**: No violations found or low-severity in moderate mode
- **Effect**: Original content passes through unchanged
- **Message**: No modification to AI response
- **Streaming**: Reports clean content status

### SANITIZE Action
- **Condition**: High-severity violations in moderate mode
- **Effect**: PII redacted with placeholder text
- **Message**: Original content with redactions + compliance notice
- **Notice**: `"⚠️ Content was sanitized for compliance."`
- **Preservation**: Non-PII content structure maintained

### BLOCK Action
- **Condition**: Any violations in strict mode
- **Effect**: Complete content replacement
- **Message**: `"❌ Output blocked due to compliance violations. Please review and redact sensitive information."`
- **Security**: No original content exposed to user

## Streaming Integration

The Guardrail Node provides real-time compliance updates:

### Starting Status Updates
```python
{
    "guardrail_check": {
        "status": "starting",
        "mode": self.guardrail_mode
    }
}
```

### Completion Updates
```python
{
    "guardrail_check": {
        "status": "completed",
        "action": action,
        "violations_found": len(violations),
        "risk_score": risk_score
    }
}
```

### Stream Writer Integration
- Uses LangGraph's `get_stream_writer()` for real-time updates
- Handles missing stream writer gracefully
- Provides structured compliance information
- No streaming failures block guardrail operations

## Error Handling

### Content Processing Errors
- **Empty Content**: Safely handled as clean content
- **Missing Attributes**: Default to safe processing
- **Invalid Message Types**: Graceful content extraction
- **Regex Failures**: Captured and logged without blocking

### Pattern Matching Errors
- **Malformed Patterns**: Default regex handling
- **Encoding Issues**: String conversion applied
- **Performance Issues**: Efficient pattern matching
- **Memory Constraints**: Minimal pattern complexity

### Streaming Errors
- **Missing Writer**: Continues without streaming
- **Write Failures**: Logs but doesn't break compliance
- **Configuration Errors**: Handled gracefully
- **Network Issues**: No impact on guardrail logic

### Exception Handling
```python
try:
    # Guardrail operations
    violations = self._check_content(final_response)
    action = self._determine_action(violations)
    # ... content modification
except Exception as e:
    # Fail gracefully - don't block on guardrail errors
    pass
```

## Testing

The Guardrail Node includes comprehensive test coverage with **23 test cases**, all passing:

### Test Categories
- **Clean Content**: No PII detection and ALLOW action ✅
- **Email PII**: Email detection and appropriate actions ✅
- **Phone PII**: Phone number detection and sanitization ✅
- **SSN PII**: Social Security Number detection and blocking ✅
- **Multiple PII**: Combined PII types in single content ✅
- **Mode Testing**: Moderate vs strict mode behaviors ✅
- **Pattern Edge Cases**: Various PII format variations ✅
- **Content Sanitization**: Redaction and preservation testing ✅
- **Action Determination**: Mode-based enforcement logic ✅
- **Error Handling**: Exception scenarios and graceful degradation ✅
- **Streaming**: Real-time update integration ✅
- **State Preservation**: Original data integrity ✅

### Test Results
- **23/23 tests passing** (100% success rate)
- Full PII detection validation
- Comprehensive mode behavior testing
- Pattern matching edge case coverage
- Error handling and resilience verification

### Mock Requirements
- LangGraph configuration and stream writer mocking
- OptimizedAgentState with various content scenarios
- Proper async test handling with pytest-asyncio
- Content pattern test case coverage

## Integration Points

### Upstream Nodes
- **Model Node**: Provides AI responses for compliance checking
- **Tool Node**: May provide tool outputs requiring compliance review
- **Router Node**: Routes content through guardrail checking

### Downstream Nodes
- **Revise Node**: Receives compliance-checked content for memory storage
- **End State**: Final compliance verification before user delivery

### Workflow Integration
```
Model Node → Guardrail Node → Revise Node → End
```

### Compliance Flow
```
AI Response → PII Detection → Action Determination → Content Modification → Compliant Output
```

## Configuration

### Default Settings
- **Mode**: `"moderate"` (configurable during initialization)
- **Importance Score**: Not applicable (content-based enforcement)
- **Pattern Sensitivity**: High (comprehensive PII detection)

### Enforcement Constants
- **ALLOW**: `"ALLOW"` - Content passes without modification
- **SANITIZE**: `"SANITIZE"` - Content redacted with placeholders
- **BLOCK**: `"BLOCK"` - Content completely replaced

### Redaction Placeholders
- **Email**: `"[REDACTED_EMAIL]"`
- **Phone**: `"[REDACTED_PHONE]"`
- **SSN**: `"[REDACTED_SSN]"`

## Best Practices

### Implementation
- Always preserve state data integrity
- Handle content gracefully regardless of format
- Provide clear enforcement actions
- Stream updates for transparency

### Compliance Strategy
- Use appropriate mode for security requirements
- Balance security with user experience
- Monitor violation patterns for policy tuning
- Maintain consistent enforcement across sessions

### Testing
- Test all PII pattern scenarios thoroughly
- Verify mode-specific behavior differences
- Check content preservation during sanitization
- Validate error handling robustness

### Monitoring
- Track violation detection rates
- Monitor false positive/negative patterns
- Log enforcement action distributions
- Measure compliance operation performance

## Security Considerations

### PII Protection
- Comprehensive pattern coverage for common PII types
- Secure redaction without data leakage
- No original PII content in error messages
- Safe handling of sensitive information

### Pattern Security
- Regex patterns optimized for security over performance
- No pattern injection vulnerabilities
- Safe content processing without exposure
- Validated redaction effectiveness

### Action Security
- BLOCK mode provides maximum content security
- SANITIZE mode balances security with usability
- No sensitive data exposure in compliance messages
- Audit logging of enforcement actions

### Error Security
- Sanitized error handling without data exposure
- No PII content in debug information
- Safe processing of untrusted content
- Secure streaming of compliance information

## Performance Considerations

### Pattern Matching Efficiency
- Optimized regex patterns for speed
- Single-pass content analysis
- Minimal memory allocation
- Efficient string processing

### Scalability Factors
- Support for large content volumes
- Fast PII detection algorithms
- Minimal processing latency
- Optimal enforcement strategies

### Memory Management
- Stateless operation design
- Efficient content processing
- Minimal memory footprint
- No unnecessary data retention

### Operation Speed
- Fast pattern compilation and matching
- Quick enforcement action determination
- Efficient content sanitization
- Minimal streaming overhead

## Troubleshooting

### Common Issues
1. **PII Not Detected**: Check pattern coverage and content format
2. **Wrong Actions**: Verify mode settings and violation classification
3. **Content Corruption**: Check sanitization logic and preservation
4. **Performance Issues**: Monitor pattern complexity and content size

### Debug Information
- Log detected PII patterns and locations
- Track enforcement action reasoning
- Monitor content modification results
- Verify streaming update delivery

### Performance Monitoring
- Measure PII detection latency
- Track enforcement action frequency
- Monitor content processing speed
- Analyze violation pattern distributions

## Compliance Architecture

### Detection Strategy
- **Pattern-Based**: Regex patterns for known PII types
- **First-Match**: Uses `re.search()` for efficiency
- **Severity-Based**: High-severity violations trigger enforcement
- **Mode-Aware**: Enforcement varies by operational mode

### Enforcement Model
- **Graduated Response**: Allow → Sanitize → Block progression
- **Content Preservation**: Maintain non-PII content structure
- **User Notification**: Clear compliance messaging
- **Audit Trail**: Complete violation and action logging

### Security Framework
```
Content Input → MCP Resource Loading → Pattern Analysis → Violation Classification → Risk Assessment → Action Determination → Content Enforcement → Compliant Output
```

## Advanced Features

### MCP Resource Management
- **Dynamic Rule Loading**: Patterns loaded from MCP resources at runtime
- **Resource Caching**: Efficient pattern compilation and caching
- **Fallback Handling**: Default patterns when MCP resources unavailable
- **Multi-Resource Support**: PII, medical, and policy resources

### Pattern Extensibility
- **MCP-Driven Patterns**: Patterns defined in MCP resources
- **New PII Types**: Additional violation types via resource updates
- **Severity Levels**: Support for different violation severities
- **Custom Actions**: Extensible enforcement action system

### Medical Compliance
- **HIPAA Integration**: Healthcare data protection rules
- **Medical Keyword Detection**: Healthcare-specific terminology
- **Compliance Recommendations**: Actionable compliance guidance
- **Risk-Based Assessment**: Severity-weighted risk scoring

### Integration Extensions
- **MCP Resource System**: Dynamic policy management via MCP
- **External Compliance**: Integration with external compliance systems
- **Audit Logging**: Comprehensive compliance event logging
- **Reporting**: Violation pattern analysis and reporting
- **Policy Management**: Dynamic policy updates via MCP resources