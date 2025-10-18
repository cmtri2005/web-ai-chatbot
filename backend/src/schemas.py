from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


class ChatConversationCreate(BaseModel):
    bot_id: str
    user_id: str
    message: str
    is_request: bool = True
    completed: bool = False


class ChatConversation(BaseModel):
    conversation_id: str
    bot_id: str
    user_id: str
    message: str
    is_request: bool
    completed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Reference(BaseModel):
    source: str = Field(
        ..., description="Tên nguồn. Ví dụ: VnExpress, Tuổi Trẻ, Luật Doanh nghiệp"
    )
    title: str = Field(
        ..., description="Tiêu đề. Ví dụ: Ngân hàng đồng loạt tăng lãi suất"
    )
    url: str = Field(..., description="URL. Ví dụ: https://vnexpress.net/...")
    content: str = Field(
        ...,
        description="Nội dung. Ví dụ: Lãi suất cho vay mua nhà hiện nay dao động từ 6-8%/năm...",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời từ AI")
    references: Optional[List[Reference]] = Field(
        default=None,
        description="Danh sách các bài báo tài chính hoặc tài liệu chính thống tham khảo (có thể bỏ trống)",
    )
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Lãi suất cho vay mua nhà hiện nay dao động từ 6-8%/năm...",
                "references": [
                    {
                        "title": "Ngân hàng đồng loạt tăng lãi suất",
                        "url": "https://vnexpress.net/...",
                        "source": "vnexpress",
                        "content": "Lãi suất cho vay mua nhà hiện nay dao động từ 6-8%/năm...",
                    }
                ],
            }
        }

