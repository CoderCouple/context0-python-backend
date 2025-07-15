#!/usr/bin/env python3
"""
Multi-Hop Reasoning Test Suite
Comprehensive test cases for validating advanced reasoning capabilities
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import aiohttp

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000/api/v1"
SAMPLE_USER_ID = "john-doe"
SAMPLE_SESSION_ID = "sample-session"


class MultiHopReasoningTester:
    """Advanced test suite for multi-hop reasoning validation"""

    def __init__(self):
        self.session = None
        self.test_results = []
        self.performance_metrics = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    def get_multihop_test_cases(self) -> List[Dict[str, Any]]:
        """Define comprehensive multi-hop reasoning test cases"""

        return [
            # LEVEL 1: Simple Connections (2-hop reasoning)
            {
                "level": 1,
                "question": "How did my childhood curiosity influence my career choice?",
                "expected_hops": 2,
                "expected_memories": 5,
                "reasoning_type": "causal_connection",
                "description": "Connect childhood trait to career outcome",
                "expected_keywords": [
                    "childhood",
                    "curiosity",
                    "electronics",
                    "computer_science",
                    "career",
                ],
            },
            {
                "level": 1,
                "question": "What connection exists between my wife Lisa and my interest in security?",
                "expected_hops": 2,
                "expected_memories": 4,
                "reasoning_type": "relationship_connection",
                "description": "Link relationship meeting to shared interests",
                "expected_keywords": [
                    "lisa",
                    "wife",
                    "tech_conference",
                    "security",
                    "mobile",
                ],
            },
            # LEVEL 2: Medium Complexity (3-4 hop reasoning)
            {
                "level": 2,
                "question": "How has mentorship shaped my journey from being mentored to becoming a mentor?",
                "expected_hops": 4,
                "expected_memories": 8,
                "reasoning_type": "temporal_progression",
                "description": "Trace mentorship from receiving to giving across time",
                "expected_keywords": [
                    "mentor",
                    "dr_chen",
                    "professor",
                    "teaching",
                    "bootcamp",
                    "engineer",
                ],
            },
            {
                "level": 2,
                "question": "What role has my family background played in my professional achievements?",
                "expected_hops": 3,
                "expected_memories": 7,
                "reasoning_type": "influence_analysis",
                "description": "Connect family influences to career success",
                "expected_keywords": [
                    "family",
                    "parents",
                    "engineer",
                    "teacher",
                    "achievement",
                    "google",
                ],
            },
            {
                "level": 2,
                "question": "How do my creative hobbies complement my technical career?",
                "expected_hops": 3,
                "expected_memories": 6,
                "reasoning_type": "complementary_analysis",
                "description": "Find synergies between different life domains",
                "expected_keywords": [
                    "guitar",
                    "photography",
                    "creative",
                    "technical",
                    "algorithm",
                    "leadership",
                ],
            },
            # LEVEL 3: Complex Reasoning (5+ hop reasoning)
            {
                "level": 3,
                "question": "How do the patterns in my relationships (family, friends, colleagues) reflect my core values and decision-making style?",
                "expected_hops": 6,
                "expected_memories": 12,
                "reasoning_type": "pattern_synthesis",
                "description": "Synthesize relationship patterns to infer values",
                "expected_keywords": [
                    "relationships",
                    "family",
                    "friends",
                    "values",
                    "patterns",
                    "decision",
                ],
            },
            {
                "level": 3,
                "question": "What evidence suggests that my technical interests, educational choices, and career progression follow a coherent narrative?",
                "expected_hops": 5,
                "expected_memories": 10,
                "reasoning_type": "narrative_coherence",
                "description": "Validate life story coherence across domains",
                "expected_keywords": [
                    "technical",
                    "education",
                    "career",
                    "narrative",
                    "coherent",
                    "progression",
                ],
            },
            {
                "level": 3,
                "question": "Based on my reflections about past experiences, what can you predict about my future decisions and priorities?",
                "expected_hops": 5,
                "expected_memories": 9,
                "reasoning_type": "predictive_analysis",
                "description": "Use meta-memories to predict future behavior",
                "expected_keywords": [
                    "reflection",
                    "future",
                    "priorities",
                    "prediction",
                    "values",
                    "growth",
                ],
            },
            # LEVEL 4: Advanced Multi-Dimensional (7+ hop reasoning)
            {
                "level": 4,
                "question": "How do the interconnections between my technical curiosity, educational mentors, career achievements, family relationships, and personal growth create a unified understanding of who I am?",
                "expected_hops": 8,
                "expected_memories": 15,
                "reasoning_type": "identity_synthesis",
                "description": "Synthesize multiple life dimensions into identity",
                "expected_keywords": [
                    "curiosity",
                    "mentors",
                    "achievements",
                    "family",
                    "growth",
                    "identity",
                ],
            },
            {
                "level": 4,
                "question": "What hidden patterns exist across my professional choices, personal relationships, hobby selections, and life reflections that reveal my unconscious decision-making framework?",
                "expected_hops": 7,
                "expected_memories": 14,
                "reasoning_type": "unconscious_pattern_detection",
                "description": "Discover hidden decision-making patterns",
                "expected_keywords": [
                    "patterns",
                    "choices",
                    "relationships",
                    "hobbies",
                    "unconscious",
                    "framework",
                ],
            },
        ]

    async def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single multi-hop reasoning test case"""
        print(f"\n🧪 LEVEL {test_case['level']} TEST: {test_case['reasoning_type']}")
        print(f"   Question: {test_case['question']}")
        print(
            f"   Expected: {test_case['expected_hops']} hops, {test_case['expected_memories']} memories"
        )

        start_time = time.time()

        payload = {
            "question": test_case["question"],
            "user_id": SAMPLE_USER_ID,
            "session_id": SAMPLE_SESSION_ID,
            "max_memories": test_case["expected_memories"] + 5,  # Allow some buffer
            "search_depth": "comprehensive",
            "include_meta_memories": True,
        }

        try:
            async with self.session.post(f"{BASE_URL}/ask", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        qa_result = result["result"]
                        reasoning_time = time.time() - start_time

                        # Analyze results
                        test_result = await self._analyze_test_result(
                            test_case, qa_result, reasoning_time
                        )
                        return test_result
                    else:
                        return self._create_failure_result(
                            test_case, f"API Error: {result.get('message')}"
                        )
                else:
                    return self._create_failure_result(
                        test_case, f"HTTP Error: {response.status}"
                    )
        except Exception as e:
            return self._create_failure_result(test_case, f"Exception: {str(e)}")

    async def _analyze_test_result(
        self,
        test_case: Dict[str, Any],
        qa_result: Dict[str, Any],
        reasoning_time: float,
    ) -> Dict[str, Any]:
        """Analyze test results for quality and correctness"""

        answer = qa_result.get("answer", "").lower()
        memories_used = qa_result.get("memories_used", 0)
        confidence = qa_result.get("confidence", 0.0)
        metadata = qa_result.get("metadata", {})

        # Check keyword coverage
        expected_keywords = test_case["expected_keywords"]
        found_keywords = [
            kw for kw in expected_keywords if kw.replace("_", " ") in answer
        ]
        keyword_coverage = len(found_keywords) / len(expected_keywords)

        # Evaluate reasoning quality
        reasoning_chains = metadata.get("reasoning_chains", 0)
        # The API returns these as counts (integers), not lists
        contradictions = metadata.get("contradictions", 0)
        gaps = metadata.get("gaps", 0)

        # Calculate score
        memory_score = min(memories_used / test_case["expected_memories"], 1.0)
        keyword_score = keyword_coverage
        confidence_score = confidence
        reasoning_score = (
            min(reasoning_chains / test_case["expected_hops"], 1.0)
            if reasoning_chains > 0
            else 0.0
        )

        overall_score = (
            memory_score + keyword_score + confidence_score + reasoning_score
        ) / 4

        # Determine pass/fail
        passed = (
            overall_score >= 0.6
            and memories_used >= test_case["expected_memories"] * 0.7
            and confidence >= 0.5
            and keyword_coverage >= 0.4
        )

        result = {
            "test_case": test_case,
            "passed": passed,
            "overall_score": overall_score,
            "metrics": {
                "memories_used": memories_used,
                "confidence": confidence,
                "reasoning_chains": reasoning_chains,
                "keyword_coverage": keyword_coverage,
                "found_keywords": found_keywords,
                "contradictions": contradictions,
                "gaps": gaps,
                "reasoning_time": reasoning_time,
                "answer_length": len(qa_result.get("answer", "")),
            },
            "answer": qa_result.get("answer", ""),
            "suggestions": qa_result.get("suggestions", []),
        }

        # Print results
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - Score: {overall_score:.2f}")
        print(
            f"   📊 Memories: {memories_used}, Confidence: {confidence:.2f}, Chains: {reasoning_chains}"
        )
        print(
            f"   🔍 Keywords: {len(found_keywords)}/{len(expected_keywords)} ({keyword_coverage:.1%})"
        )
        print(f"   💬 Answer: {qa_result.get('answer', '')[:100]}...")

        return result

    def _create_failure_result(
        self, test_case: Dict[str, Any], error_message: str
    ) -> Dict[str, Any]:
        """Create failure result for failed test cases"""
        print(f"   ❌ FAIL - {error_message}")

        return {
            "test_case": test_case,
            "passed": False,
            "overall_score": 0.0,
            "error": error_message,
            "metrics": {},
            "answer": "",
            "suggestions": [],
        }

    async def test_reasoning_explanation_depth(self):
        """Test the depth and quality of reasoning explanations"""
        print(f"\n🔍 Testing Reasoning Explanation Depth...")

        complex_question = "How do my professional achievements, personal relationships, and creative hobbies reveal a pattern of seeking both technical excellence and human connection?"

        payload = {
            "question": complex_question,
            "user_id": SAMPLE_USER_ID,
            "session_id": SAMPLE_SESSION_ID,
            "max_memories": 15,
            "search_depth": "comprehensive",
        }

        async with self.session.post(
            f"{BASE_URL}/explain-reasoning", json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("success"):
                    reasoning_data = result["result"]

                    print(f"   📝 Question: {reasoning_data['question']}")
                    print(
                        f"   🔗 Reasoning Chains: {len(reasoning_data['reasoning_chains'])}"
                    )

                    total_steps = sum(
                        len(chain["steps"])
                        for chain in reasoning_data["reasoning_chains"]
                    )
                    avg_confidence = sum(
                        chain["overall_confidence"]
                        for chain in reasoning_data["reasoning_chains"]
                    ) / len(reasoning_data["reasoning_chains"])

                    print(f"   📊 Total Reasoning Steps: {total_steps}")
                    print(f"   🎯 Average Chain Confidence: {avg_confidence:.2f}")
                    print(
                        f"   ⚠️  Contradictions Found: {len(reasoning_data['contradictions'])}"
                    )
                    print(f"   🔍 Information Gaps: {len(reasoning_data['gaps'])}")

                    # Show first reasoning chain details
                    if reasoning_data["reasoning_chains"]:
                        first_chain = reasoning_data["reasoning_chains"][0]
                        print(
                            f"   \n   🧠 Sample Reasoning Chain (Confidence: {first_chain['overall_confidence']:.2f}):"
                        )
                        for i, step in enumerate(
                            first_chain["steps"][:3]
                        ):  # Show first 3 steps
                            print(f"     Step {i+1}: {step['step_type']}")
                            print(f"       {step['reasoning_process'][:80]}...")

                    print(f"   ✅ Reasoning explanation test completed")
                else:
                    print(f"   ❌ Failed: {result.get('message')}")
            else:
                print(f"   ❌ HTTP Error {response.status}")

    async def run_comprehensive_test_suite(self):
        """Run all multi-hop reasoning tests"""
        print("🚀 Multi-Hop Reasoning Test Suite")
        print("=" * 80)

        test_cases = self.get_multihop_test_cases()
        all_results = []

        # Run each test case
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}]", end="")
            result = await self.run_test_case(test_case)
            all_results.append(result)
            self.test_results.append(result)

            # Brief pause between tests
            await asyncio.sleep(0.5)

        # Test reasoning explanation depth
        await self.test_reasoning_explanation_depth()

        # Generate comprehensive report
        self._generate_test_report(all_results)

        return all_results

    def _generate_test_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive test report"""
        print(f"\n" + "=" * 80)
        print("📊 MULTI-HOP REASONING TEST REPORT")
        print("=" * 80)

        # Overall statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r["passed"])
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0

        print(f"📈 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Pass Rate: {pass_rate:.1%}")

        # Results by level
        level_stats = {}
        for result in results:
            level = result["test_case"]["level"]
            if level not in level_stats:
                level_stats[level] = {"total": 0, "passed": 0}
            level_stats[level]["total"] += 1
            if result["passed"]:
                level_stats[level]["passed"] += 1

        print(f"\n📊 Results by Complexity Level:")
        for level in sorted(level_stats.keys()):
            stats = level_stats[level]
            level_pass_rate = (
                stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            )
            print(
                f"   Level {level}: {stats['passed']}/{stats['total']} ({level_pass_rate:.1%})"
            )

        # Performance metrics
        successful_results = [r for r in results if r["passed"] and "metrics" in r]
        if successful_results:
            avg_memories = sum(
                r["metrics"]["memories_used"] for r in successful_results
            ) / len(successful_results)
            avg_confidence = sum(
                r["metrics"]["confidence"] for r in successful_results
            ) / len(successful_results)
            avg_reasoning_time = sum(
                r["metrics"]["reasoning_time"] for r in successful_results
            ) / len(successful_results)
            avg_chains = sum(
                r["metrics"]["reasoning_chains"] for r in successful_results
            ) / len(successful_results)

            print(f"\n🎯 Performance Metrics (Successful Tests):")
            print(f"   Average Memories Used: {avg_memories:.1f}")
            print(f"   Average Confidence: {avg_confidence:.2f}")
            print(f"   Average Reasoning Time: {avg_reasoning_time:.2f}s")
            print(f"   Average Reasoning Chains: {avg_chains:.1f}")

        # Failed test analysis
        failed_results = [r for r in results if not r["passed"]]
        if failed_results:
            print(f"\n❌ Failed Test Analysis:")
            for result in failed_results:
                test_case = result["test_case"]
                reason = result.get("error", "Low performance metrics")
                print(
                    f"   Level {test_case['level']}: {test_case['reasoning_type']} - {reason}"
                )

        # Recommendations
        print(f"\n💡 Recommendations:")
        if pass_rate >= 0.8:
            print(f"   ✅ Excellent multi-hop reasoning performance!")
            print(f"   🎯 System handles complex reasoning across multiple domains")
        elif pass_rate >= 0.6:
            print(f"   ⚠️  Good performance with room for improvement")
            print(f"   🔧 Consider tuning confidence thresholds and memory clustering")
        else:
            print(f"   ❌ Performance needs improvement")
            print(
                f"   🔧 Review reasoning chain logic and memory aggregation algorithms"
            )

        print(f"\n✅ Test report completed!")


async def main():
    """Main test runner"""
    print("🧪 Multi-Hop Reasoning Validation Test Suite")
    print("Testing advanced reasoning capabilities with sample data")
    print()

    # Check server
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/ping") as response:
                if response.status != 200:
                    print(
                        "❌ Server not running. Start with: poetry run uvicorn app.main:app --reload"
                    )
                    return
                print("✅ Server is running")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return

    # Check sample data exists
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "question": "Tell me about my technical skills and expertise",
                "user_id": SAMPLE_USER_ID,
                "session_id": SAMPLE_SESSION_ID,
                "max_memories": 3,
            }
            async with session.post(f"{BASE_URL}/ask", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success") and result["result"]["memories_used"] > 0:
                        print("✅ Sample data detected")
                    else:
                        print(
                            "❌ No sample data found. Run: python create_sample_data.py"
                        )
                        return
                else:
                    print("❌ Cannot verify sample data")
                    return
    except Exception as e:
        print(f"❌ Sample data verification failed: {e}")
        return

    # Run tests
    async with MultiHopReasoningTester() as tester:
        await tester.run_comprehensive_test_suite()


if __name__ == "__main__":
    asyncio.run(main())
