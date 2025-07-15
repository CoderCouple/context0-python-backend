"""Q&A API Controller with multi-hop reasoning"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from starlette import status

from app.api.tags import Tags
from app.api.v1.request.qa_request import ConversationRequest, QuestionRequest
from app.api.v1.response.base_response import (
    BaseResponse,
    error_response,
    success_response,
)
from app.api.v1.response.qa_response import (
    ConversationResponse,
    QAHealthResponse,
    QuestionResponse,
)
from app.service.qa_service import QAService

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.QA])


# Dependency injection for QA service
async def get_qa_service() -> QAService:
    """Get QA service instance"""
    return QAService()


@router.post("/ask", response_model=BaseResponse[QuestionResponse])
async def ask_question(
    request: QuestionRequest,
    background_tasks: BackgroundTasks,
    qa_service: QAService = Depends(get_qa_service),
):
    """Ask a question based on stored memories with multi-hop reasoning"""
    try:
        logger.info(
            f"Processing question for user {request.user_id}: {request.question}"
        )

        # Process the question using multi-hop reasoning
        response = await qa_service.ask_question(request)

        # Add background logging
        background_tasks.add_task(
            _log_qa_interaction,
            request.question,
            response.answer,
            request.user_id,
            response.confidence,
        )

        return success_response(
            result=response, message="Question processed successfully", status_code=200
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return error_response(
            message=f"Failed to process question: {str(e)}", status_code=500
        )


@router.post("/conversation", response_model=BaseResponse[ConversationResponse])
async def conversation(
    request: ConversationRequest, qa_service: QAService = Depends(get_qa_service)
):
    """Have a conversation with memory-based context"""
    try:
        logger.info(f"Processing conversation for user {request.user_id}")

        # Extract the latest message as the question
        if not request.messages:
            raise HTTPException(
                status_code=400, detail="No messages provided in conversation request"
            )

        latest_message = request.messages[-1]
        if latest_message.get("role") != "user":
            raise HTTPException(
                status_code=400, detail="Latest message must be from user"
            )

        # Convert to QuestionRequest for processing
        question_request = QuestionRequest(
            question=latest_message["content"],
            user_id=request.user_id,
            session_id=request.session_id,
            max_memories=request.max_memories,
            search_depth="comprehensive",
        )

        # Get the answer
        qa_response = await qa_service.ask_question(question_request)

        # Convert to conversation response
        conversation_response = ConversationResponse(
            response=qa_response.answer,
            confidence=qa_response.confidence,
            context_memories=qa_response.memory_contexts,
            conversation_context=request.messages[
                -request.conversation_context_window :
            ],
            follow_up_suggestions=qa_response.suggestions,
            processing_time_ms=qa_response.processing_time_ms,
        )

        return success_response(
            result=conversation_response, message="Conversation processed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing conversation: {e}")
        return error_response(
            message=f"Failed to process conversation: {str(e)}", status_code=500
        )


@router.get("/qa/health", response_model=BaseResponse[QAHealthResponse])
async def health_check(qa_service: QAService = Depends(get_qa_service)):
    """Check the health of the Q&A system"""
    try:
        health_status = await qa_service.health_check()

        status_code = 200 if health_status.status == "healthy" else 503

        return success_response(
            result=health_status,
            message=f"Q&A system status: {health_status.status}",
            status_code=status_code,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return error_response(message=f"Health check failed: {str(e)}", status_code=500)


@router.get("/stats")
async def get_qa_stats(user_id: str, qa_service: QAService = Depends(get_qa_service)):
    """Get Q&A statistics for a user"""
    try:
        # This would return statistics about Q&A usage
        # For now, return basic info
        return success_response(
            result={
                "user_id": user_id,
                "total_questions": 0,  # Would be tracked in audit logs
                "average_confidence": 0.0,
                "memory_stores_available": [
                    "document_store",
                    "graph_store",
                    "vector_store",
                    "timeseries_store",
                    "audit_store",
                ],
                "reasoning_capabilities": [
                    "semantic_search",
                    "multi_hop_reasoning",
                    "cross_database_references",
                    "temporal_analysis",
                    "relationship_traversal",
                ],
            },
            message="Q&A statistics retrieved",
        )

    except Exception as e:
        logger.error(f"Error getting Q&A stats: {e}")
        return error_response(
            message=f"Failed to get Q&A stats: {str(e)}", status_code=500
        )


@router.post("/explain-reasoning")
async def explain_reasoning(
    request: QuestionRequest, qa_service: QAService = Depends(get_qa_service)
):
    """Explain the reasoning process for a question"""
    try:
        # Get the reasoning result with full details
        reasoning_result = await qa_service.reasoning_engine.reason_about_question(
            request.question, request.user_id
        )

        return success_response(
            result={
                "question": reasoning_result.question,
                "reasoning_chains": [
                    {
                        "chain_id": chain.chain_id,
                        "steps": [
                            {
                                "step_type": step.step_type,
                                "reasoning_process": step.reasoning_process,
                                "output": step.output,
                                "confidence": step.confidence,
                            }
                            for step in chain.reasoning_steps
                        ],
                        "overall_confidence": chain.overall_confidence,
                    }
                    for chain in [reasoning_result.primary_chain]
                    + reasoning_result.alternative_chains
                ],
                "synthesis": reasoning_result.synthesis,
                "contradictions": reasoning_result.contradictions,
                "gaps": reasoning_result.gaps,
                "metadata": reasoning_result.reasoning_metadata,
            },
            message="Reasoning explanation generated",
        )

    except Exception as e:
        logger.error(f"Error explaining reasoning: {e}")
        return error_response(
            message=f"Failed to explain reasoning: {str(e)}", status_code=500
        )


async def _log_qa_interaction(
    question: str, answer: str, user_id: str, confidence: float
):
    """Background task to log Q&A interactions"""
    try:
        # This would log the interaction to audit store or analytics system
        logger.info(f"Q&A Interaction - User: {user_id}, Confidence: {confidence:.2f}")
    except Exception as e:
        logger.error(f"Failed to log Q&A interaction: {e}")
