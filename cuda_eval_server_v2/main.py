"""
Main entry point for CUDA Evaluation Server V2
Handles argument parsing and server startup with integrated MCP support
"""

import argparse
import logging
import os
import sys
import uvicorn

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="CUDA Evaluation Server V2 - Unified FastAPI + MCP Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port number to bind to")
    parser.add_argument("--log-level", type=str, default="info",
                       choices=["debug", "info", "warning", "error", "critical"],
                       help="Log level")
    parser.add_argument("--cupy-cache-dir", type=str, default="/tmp/cupy_kernel_cache",
                       help="Directory for CuPy kernel cache")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    # Set environment variables
    os.environ["CUPY_KERNEL_CACHE_DIR"] = args.cupy_cache_dir

    # Log startup info
    logger.info("🚀 Starting CUDA Evaluation Server V2 (Unified FastAPI + MCP)")
    logger.info(f"   Host: {args.host}:{args.port}")
    logger.info(f"   CuPy Cache Dir: {args.cupy_cache_dir}")
    logger.info(f"   Log Level: {args.log_level.upper()}")
    logger.info("")
    logger.info("   Available Endpoints:")
    logger.info(f"   📡 FastAPI REST API:")
    logger.info(f"      - Health Check:  http://{args.host}:{args.port}/health")
    logger.info(f"      - Evaluate:      http://{args.host}:{args.port}/evaluate")
    logger.info(f"      - Compare:       http://{args.host}:{args.port}/compare")
    logger.info(f"   🤖 MCP (Model Context Protocol):")
    logger.info(f"      - MCP Endpoint:  http://{args.host}:{args.port}/mcp")
    logger.info("")
    logger.info("   Technology Stack: FastAPI + CuPy + Async + MCP (Streamable HTTP)")

    if args.reload:
        logger.info("   ⚠️  Development mode: Auto-reload enabled")

    try:
        # Run unified server (FastAPI with mounted MCP)
        uvicorn.run(
            "app:app",
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            reload=args.reload,
            access_log=True
        )

    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        import traceback
        logger.error(f"❌ Server failed to start: {e}")
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
