# Test Suite for Multi-Hop Reasoning System

This directory contains comprehensive test scripts for the multi-hop reasoning memory system.

## Test Scripts Overview

### Setup & Data Management
- **`flush_databases.py`** - Clean all databases for fresh start
- **`create_sample_data.py`** - Create comprehensive sample data for testing

### Core Testing
- **`test_qa_system.py`** - Comprehensive Q&A API tests
- **`demo_qa.py`** - Simple demonstration script
- **`test_multihop_reasoning.py`** - Advanced multi-hop reasoning validation
- **`test_large_scale_reasoning.py`** - Large-scale memory cross-referencing (20-50+ memories)

### Legacy Tests
- **`test_api_call.py`** - Basic API call tests
- **`test_config.py`** - Configuration tests
- **`test_detailed_config.py`** - Detailed configuration validation
- **`test_memory_api.py`** - Memory API specific tests

## Testing Steps

### 1. Fresh Start (Optional)
```bash
python tests/flush_databases.py
```

### 2. Start Server
```bash
poetry run uvicorn app.main:app --reload
```

### 3. Create Sample Data
```bash
python tests/create_sample_data.py
```

### 4. Run Tests

**Quick Demo:**
```bash
python tests/demo_qa.py
```

**Comprehensive Testing:**
```bash
python tests/test_qa_system.py
```

**Multi-Hop Reasoning Validation:**
```bash
python tests/test_multihop_reasoning.py
```

**Large-Scale Memory Testing:**
```bash
python tests/test_large_scale_reasoning.py
```

## Test Validation Levels

### Level 1: Simple Connections (2-hop)
- Basic causal relationships
- Direct memory connections
- Expected: 5 memories, 80%+ confidence

### Level 2: Medium Complexity (3-4 hop)
- Temporal progressions
- Influence analysis
- Expected: 6-8 memories, 70%+ confidence

### Level 3: Complex Reasoning (5+ hop)
- Pattern synthesis
- Narrative coherence
- Predictive analysis
- Expected: 9-12 memories, 60%+ confidence

### Level 4: Advanced Multi-Dimensional (7+ hop)
- Identity synthesis
- Unconscious pattern detection
- Expected: 13-15 memories, 50%+ confidence

## Expected Performance

- **Pass Rate**: 80%+ for comprehensive reasoning
- **Memory Usage**: 5-15 memories per query
- **Response Time**: <3 seconds per query
- **Cross-Database Integration**: All 5 stores (Vector, Document, Graph, TimeSeries, Audit)

## Key Features Tested

✅ Cross-database memory references  
✅ Multi-hop reasoning across memory types  
✅ Temporal and relationship analysis  
✅ Meta-memory and reflection processing  
✅ Conversational context handling  
✅ Reasoning explanation and transparency  
✅ Large-scale memory clustering  
✅ Pattern recognition across life domains  

## Sample User

All tests use sample user `john-doe` with comprehensive life story including:
- Personal background and family
- Education journey (University of Washington, CS)
- Career progression (Amazon → Google)
- Relationships (wife Lisa, friend Alex, sister Emma)
- Hobbies (guitar, hiking, photography)
- Reflections and growth patterns

The sample data is specifically designed to test multi-hop reasoning capabilities across different life domains and time periods.