"""
多模态处理器 - 支持语音转录、图像分析和文档处理
"""
import os
import tempfile
import asyncio
from typing import Dict, List, Optional, Any
from fastapi import UploadFile

from isa_agent_sdk.utils.logger import api_logger
from .storage_service import get_storage_service


class MultimodalProcessor:
    """
    多模态文件处理器
    支持语音转录、图像分析、文档处理等功能
    """
    
    def __init__(self, isa_url: str = None, storage_service_url: Optional[str] = None):
        if isa_url is None:
            from isa_agent_sdk.core.config import settings
            isa_url = settings.resolved_isa_api_url
        """初始化多模态处理器 - 使用Consul服务发现"""
        self.temp_files = []
        self.isa_url = isa_url
        # Use Consul service discovery for storage service
        self.storage_service = get_storage_service(storage_service_url)
    async def close(self):
        """关闭资源并清理临时文件"""
        # 清理临时文件
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                api_logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        self.temp_files.clear()
    
    async def process_audio_file(self, audio: UploadFile) -> Dict[str, Any]:
        """
        处理语音文件 - 使用ISA Model Service进行转录
        
        Args:
            audio: 上传的音频文件
            
        Returns:
            包含转录文本和元数据的字典
        """
        try:
            # 检查文件大小 (25MB limit for OpenAI Whisper)
            content = await audio.read()
            if len(content) > 25 * 1024 * 1024:
                return {
                    "text": "",
                    "error": "Audio file too large (max 25MB)",
                    "success": False
                }
            
            # 重置文件指针
            await audio.seek(0)
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=self._get_file_extension(audio.filename)
            )
            temp_file.write(content)
            temp_file.close()
            self.temp_files.append(temp_file.name)
            
            api_logger.info(f"🎤 Processing audio file: {audio.filename} ({len(content)} bytes)")
            
            # 使用ISA客户端进行音频转录
            from isa_model import ISAModelClient
            client = ISAModelClient()
            result = await client.invoke(
                content,  # 使用bytes数据
                "transcribe", 
                "audio",
                filename=audio.filename  # 提供原始文件名
            )
            
            if result.get("success"):
                transcription_result = result.get("result", {})
                transcribed_text = transcription_result.get("text", "")
                
                api_logger.info(f"✅ Audio transcribed successfully: {len(transcribed_text)} characters")
                
                return {
                    "text": transcribed_text,
                    "language": transcription_result.get("language", "unknown"),
                    "duration": transcription_result.get("duration"),
                    "confidence": transcription_result.get("confidence"),
                    "segments": transcription_result.get("segments", []),
                    "success": True,
                    "filename": audio.filename,
                    "file_size": len(content),
                    "model_used": result.get("metadata", {}).get("model_used", "whisper-1")
                }
            else:
                error_msg = result.get("error", "Unknown error")
                api_logger.error(f"❌ Audio transcription failed: {error_msg}")
                
                return {
                    "text": "",
                    "error": f"Audio transcription failed: {error_msg}",
                    "success": False
                }
                        
        except Exception as e:
            api_logger.error(f"❌ Audio processing failed: {str(e)}")
            return {
                "text": "",
                "error": f"Audio processing failed: {str(e)}",
                "success": False
            }
    
    async def process_files(self, files: List[UploadFile]) -> List[Dict[str, Any]]:
        """
        处理多个文件
        
        Args:
            files: 上传的文件列表
            
        Returns:
            处理结果列表
        """
        results = []
        
        for file in files:
            try:
                content = await file.read()
                await file.seek(0)
                
                file_info = {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(content),
                    "content": "",
                    "success": True
                }
                
                # 根据文件类型进行不同处理
                if file.content_type and file.content_type.startswith('text/'):
                    # 文本文件直接读取
                    file_info["content"] = content.decode('utf-8', errors='ignore')
                    api_logger.info(f"📄 Text file processed: {file.filename}")
                    
                elif file.content_type and file.content_type.startswith('image/'):
                    # 图像文件处理 - 使用ISA模型服务进行图像分析
                    try:
                        from isa_model import ISAModelClient
                        client = ISAModelClient()
                        
                        # 使用ISA服务分析图像 - 使用正确的参数格式
                        image_result = await client.invoke(
                            input_data=content,  # 图像字节数据
                            task="analyze",  # 分析任务
                            service_type="vision",  # 视觉服务
                            filename=file.filename
                        )
                        
                        if image_result.get("success"):
                            analysis = image_result.get("result", {})
                            # 更详细地处理结果，尝试不同的可能字段
                            description = (analysis.get('description') or 
                                         analysis.get('text') or 
                                         analysis.get('analysis') or 
                                         str(analysis) if analysis else '无法获取分析结果')
                            file_info["content"] = f"[Image: {file.filename}]\n图像分析结果: {description}"
                            api_logger.info(f"🖼️ Image analyzed successfully: {file.filename}")
                            api_logger.info(f"🖼️ Analysis result: {description[:200]}...")
                        else:
                            file_info["content"] = f"[Image: {file.filename}, {len(content)} bytes]\n图像分析失败，但文件已上传"
                            api_logger.warning(f"🖼️ Image analysis failed for {file.filename}: {image_result.get('error')}")
                            
                    except Exception as e:
                        # 回退到基本处理
                        file_info["content"] = f"[Image: {file.filename}, {len(content)} bytes]\n图像处理异常: {str(e)}"
                        api_logger.warning(f"🖼️ Image processing exception for {file.filename}: {e}")
                    
                elif file.filename and file.filename.lower().endswith('.pdf'):
                    # PDF文件处理
                    try:
                        from isa_model import ISAModelClient
                        client = ISAModelClient()
                        
                        # 使用ISA服务处理PDF - 使用正确的参数格式
                        pdf_result = await client.invoke(
                            input_data=content,  # PDF字节数据
                            task="extract_text",  # 文本提取任务
                            service_type="document",  # 文档服务
                            filename=file.filename
                        )
                        
                        if pdf_result.get("success"):
                            extracted_text = pdf_result.get("result", {}).get("text", "")
                            # 限制文本长度避免过长
                            if len(extracted_text) > 2000:
                                extracted_text = extracted_text[:2000] + "...(内容截断)"
                            file_info["content"] = f"[PDF: {file.filename}]\n文档内容:\n{extracted_text}"
                            api_logger.info(f"📄 PDF processed successfully: {file.filename}")
                        else:
                            file_info["content"] = f"[PDF: {file.filename}, {len(content)} bytes]\nPDF处理失败，但文件已上传"
                            api_logger.warning(f"📄 PDF processing failed for {file.filename}")
                            
                    except Exception as e:
                        # 回退到基本处理
                        file_info["content"] = f"[PDF: {file.filename}, {len(content)} bytes]\nPDF处理异常: {str(e)}"
                        api_logger.warning(f"📄 PDF processing exception for {file.filename}: {e}")
                    
                else:
                    # 其他文件类型
                    file_info["content"] = f"[File: {file.filename}, {len(content)} bytes, type: {file.content_type}]"
                    api_logger.info(f"📎 File processed: {file.filename}")
                
                results.append(file_info)
                
            except Exception as e:
                api_logger.error(f"❌ File processing failed for {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "content": "",
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    async def process_files_with_storage(
        self, 
        files: List[UploadFile], 
        user_id: str, 
        auth_token: Optional[str] = None,
        organization_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process files and upload to storage service with automatic RAG indexing
        
        Args:
            files: Uploaded files
            user_id: User ID
            auth_token: Authentication token
            organization_id: Organization ID (optional)
            
        Returns:
            Processing results with file IDs and storage information
        """
        if not files:
            return {
                'success': False,
                'error': 'No files provided'
            }
        
        uploaded_files = []
        processing_summary = []
        
        try:
            for file in files:
                # Read file content
                file_content = await file.read()
                await file.seek(0)
                
                # Process file for immediate use
                processed_info = {
                    'filename': file.filename,
                    'content_type': file.content_type,
                    'size': len(file_content)
                }
                
                # Process based on file type for immediate response
                if file.content_type and file.content_type.startswith('text/'):
                    processed_info['content'] = file_content.decode('utf-8', errors='ignore')
                    processed_info['type'] = 'text'
                    
                elif file.content_type and file.content_type.startswith('image/'):
                    try:
                        from isa_model import ISAModelClient
                        client = ISAModelClient()
                        
                        image_result = await client.invoke(
                            input_data=file_content,
                            task="analyze",
                            service_type="vision",
                            filename=file.filename
                        )
                        
                        if image_result.get("success"):
                            analysis = image_result.get("result", {})
                            description = (analysis.get('description') or 
                                         analysis.get('text') or 
                                         analysis.get('analysis') or 
                                         str(analysis) if analysis else 'Image analysis completed')
                            processed_info['content'] = f"[Image Analysis]: {description}"
                            processed_info['type'] = 'image'
                        else:
                            processed_info['content'] = f"[Image]: {file.filename} (analysis failed but uploaded)"
                            processed_info['type'] = 'image'
                            
                    except Exception as e:
                        processed_info['content'] = f"[Image]: {file.filename} (processing error: {str(e)})"
                        processed_info['type'] = 'image'
                    
                elif file.filename and file.filename.lower().endswith('.pdf'):
                    try:
                        from isa_model import ISAModelClient
                        client = ISAModelClient()
                        
                        pdf_result = await client.invoke(
                            input_data=file_content,
                            task="extract_text",
                            service_type="document",
                            filename=file.filename
                        )
                        
                        if pdf_result.get("success"):
                            extracted_text = pdf_result.get("result", {}).get("text", "")
                            if len(extracted_text) > 1000:
                                extracted_text = extracted_text[:1000] + "...(content continues)"
                            processed_info['content'] = f"[PDF Content]: {extracted_text}"
                            processed_info['type'] = 'pdf'
                        else:
                            processed_info['content'] = f"[PDF]: {file.filename} (text extraction failed but uploaded)"
                            processed_info['type'] = 'pdf'
                            
                    except Exception as e:
                        processed_info['content'] = f"[PDF]: {file.filename} (processing error: {str(e)})"
                        processed_info['type'] = 'pdf'
                    
                else:
                    processed_info['content'] = f"[File]: {file.filename} (uploaded successfully)"
                    processed_info['type'] = 'other'
                
                # Upload to storage service WITHOUT automatic indexing
                # We don't want auto-indexing for any files uploaded via chat
                # Let the graph/tools handle processing based on file type
                try:
                    upload_result = self.storage_service.upload_file(
                        file_content=file_content,
                        filename=file.filename,
                        content_type=file.content_type or 'application/octet-stream',
                        user_id=user_id,
                        organization_id=organization_id,
                        metadata={
                            'processed_type': processed_info['type'],
                            'uploaded_via': 'multimodal_chat',
                            'size': len(file_content)
                        },
                        tags=['chat_upload', processed_info['type']],
                        auth_token=auth_token,
                        enable_indexing=False  # Never auto-index, let tools handle it
                    )
                    
                    if upload_result.get('success'):
                        processed_info['file_id'] = upload_result.get('file_id')
                        processed_info['download_url'] = upload_result.get('download_url')
                        processed_info['storage_success'] = True
                        api_logger.info(f"File uploaded and indexed: {file.filename} -> {processed_info['file_id']}")
                    else:
                        processed_info['storage_success'] = False
                        processed_info['storage_error'] = upload_result.get('error', 'Unknown storage error')
                        api_logger.warning(f"Storage upload failed for {file.filename}: {processed_info['storage_error']}")
                    
                except Exception as e:
                    processed_info['storage_success'] = False
                    processed_info['storage_error'] = f"Storage service error: {str(e)}"
                    api_logger.error(f"Storage service error for {file.filename}: {e}")
                
                uploaded_files.append(processed_info)
                
                # Create summary for chat
                if processed_info.get('storage_success'):
                    summary = f"✅ {file.filename}: {processed_info['content'][:200]}{'...' if len(processed_info['content']) > 200 else ''}"
                else:
                    summary = f"⚠️ {file.filename}: {processed_info['content'][:200]}{'...' if len(processed_info['content']) > 200 else ''} (storage failed)"
                
                processing_summary.append(summary)
            
            # Create combined response
            combined_content = "\n\n".join(processing_summary)
            
            return {
                'success': True,
                'files': uploaded_files,
                'combined_content': combined_content,
                'files_processed': len(uploaded_files),
                'files_stored': sum(1 for f in uploaded_files if f.get('storage_success', False)),
                'message': f"Processed {len(uploaded_files)} files, {sum(1 for f in uploaded_files if f.get('storage_success', False))} stored with RAG indexing"
            }
            
        except Exception as e:
            api_logger.error(f"Error in process_files_with_storage: {e}")
            return {
                'success': False,
                'error': f"File processing error: {str(e)}",
                'files': uploaded_files,
                'files_processed': len(uploaded_files)
            }
    
    def _get_file_extension(self, filename: Optional[str]) -> str:
        """获取文件扩展名"""
        if not filename:
            return '.mp3'  # 默认音频格式
        
        ext = os.path.splitext(filename)[1].lower()
        if not ext:
            return '.mp3'
        
        # 支持的音频格式
        supported_audio = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm']
        if ext in supported_audio:
            return ext
        else:
            return '.mp3'  # 默认转换为mp3