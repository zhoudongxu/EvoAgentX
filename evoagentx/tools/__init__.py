"""Load optional tool integrations only when explicitly requested."""

from importlib import import_module
from .tool import Tool, Toolkit

_EXPORTS = {
    "DockerInterpreterToolkit": ("interpreter_docker", "DockerInterpreterToolkit"),
    "PythonInterpreterToolkit": ("interpreter_python", "PythonInterpreterToolkit"),
    "GoogleSearchToolkit": ("search_google", "GoogleSearchToolkit"),
    "GoogleFreeSearchToolkit": ("search_google_f", "GoogleFreeSearchToolkit"),
    "DDGSSearchToolkit": ("search_ddgs", "DDGSSearchToolkit"),
    "WikipediaSearchToolkit": ("search_wiki", "WikipediaSearchToolkit"),
    "BrowserToolkit": ("browser_tool", "BrowserToolkit"),
    "MCPToolkit": ("mcp", "MCPToolkit"),
    "RequestToolkit": ("request", "RequestToolkit"),
    "ArxivToolkit": ("request_arxiv", "ArxivToolkit"),
    "BrowserUseToolkit": ("browser_use", "BrowserUseToolkit"),
    "GoogleMapsToolkit": ("google_maps_tool", "GoogleMapsToolkit"),
    "TelegramToolkit": ("telegram_tools", "TelegramToolkit"),
    "GmailToolkit": ("gmail_tools", "GmailToolkit"),
    "MongoDBToolkit": ("database_mongodb", "MongoDBToolkit"),
    "PostgreSQLToolkit": ("database_postgresql", "PostgreSQLToolkit"),
    "FileStorageHandler": ("storage_handler", "FileStorageHandler"),
    "LocalStorageHandler": ("storage_handler", "LocalStorageHandler"),
    "SupabaseStorageHandler": ("storage_handler", "SupabaseStorageHandler"),
    "StorageToolkit": ("storage_file", "StorageToolkit"),
    "FluxImageGenerationEditTool": (
        "image_tools.flux_image_tools.image_generation_edit",
        "FluxImageGenerationEditTool",
    ),
    "FluxImageGenerationToolkit": (
        "image_tools.flux_image_tools.toolkit",
        "FluxImageGenerationToolkit",
    ),
    "OpenAIImageToolkit": (
        "image_tools.openai_image_tools.toolkit",
        "OpenAIImageToolkit",
    ),
    "OpenRouterImageAnalysisTool": (
        "image_tools.openrouter_image_tools.image_analysis",
        "ImageAnalysisTool",
    ),
    "OpenRouterImageGenerationEditTool": (
        "image_tools.openrouter_image_tools.image_generation",
        "OpenRouterImageGenerationEditTool",
    ),
    "OpenRouterImageToolkit": (
        "image_tools.openrouter_image_tools.toolkit",
        "OpenRouterImageToolkit",
    ),
    "CMDToolkit": ("cmd_toolkit", "CMDToolkit"),
    "RSSToolkit": ("rss_feed", "RSSToolkit"),
    "FileToolkit": ("file_tool", "FileToolkit"),
    "SerperAPIToolkit": ("search_serperapi", "SerperAPIToolkit"),
    "SerpAPIToolkit": ("search_serpapi", "SerpAPIToolkit"),
    "ExaSearchToolkit": ("search_exa", "ExaSearchToolkit"),
    "ResearchToolkit": ("research_tools", "ResearchToolkit"),
}
__all__ = ["Tool", "Toolkit", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    try:
        value = getattr(import_module("." + module, __name__), attribute)
    except ImportError:
        if name != "ResearchToolkit":
            raise
        value = None
    globals()[name] = value
    return value
