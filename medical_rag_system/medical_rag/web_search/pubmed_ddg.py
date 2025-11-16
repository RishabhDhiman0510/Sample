"""Async web search with retry and caching."""
import asyncio, aiohttp, hashlib, json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ..utils.logging import get_logger

logger = get_logger(__name__)

class WebSearchCache:
    def __init__(self, cache_dir: str = "./.web_cache", ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _get_cache_key(self, query: str, source: str) -> str:
        key_str = f"{source}:{query}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, source: str) -> Optional[List[str]]:
        cache_key = self._get_cache_key(query, source)
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            cached_time = datetime.fromisoformat(data["timestamp"])
            if datetime.now() - cached_time > timedelta(seconds=self.ttl):
                logger.info("cache_expired", query=query, source=source)
                return None
            logger.info("cache_hit", query=query, source=source)
            return data["results"]
        except Exception as e:
            logger.warning("cache_read_error", error=str(e))
            return None

    def set(self, query: str, source: str, results: List[str]) -> None:
        cache_key = self._get_cache_key(query, source)
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            data = {"timestamp": datetime.now().isoformat(), "query": query, "source": source, "results": results}
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            logger.info("cache_set", query=query, source=source)
        except Exception as e:
            logger.warning("cache_write_error", error=str(e))

class PubMedSearch:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def __init__(self, email: Optional[str] = None, timeout: int = 5, cache: Optional[WebSearchCache] = None):
        self.email = email
        self.timeout = timeout
        self.cache = cache

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
    async def _fetch_with_retry(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        async with session.get(url, timeout=self.timeout) as response:
            response.raise_for_status()
            return await response.json()

    async def search(self, query: str, max_results: int = 3) -> List[str]:
        if self.cache:
            cached = self.cache.get(query, "pubmed")
            if cached is not None:
                return cached
        logger.info("pubmed_search_start", query=query)
        try:
            async with aiohttp.ClientSession() as session:
                search_url = f"{self.BASE_URL}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
                if self.email:
                    search_url += f"&email={self.email}"
                search_data = await self._fetch_with_retry(session, search_url)
                pmids = search_data.get('esearchresult', {}).get('idlist', [])
                if not pmids:
                    logger.info("pubmed_no_results", query=query)
                    return []
                fetch_url = f"{self.BASE_URL}esummary.fcgi?db=pubmed&id={','.join(pmids)}&retmode=json"
                if self.email:
                    fetch_url += f"&email={self.email}"
                fetch_data = await self._fetch_with_retry(session, fetch_url)
                results = []
                for pmid in pmids:
                    if pmid in fetch_data.get('result', {}):
                        article = fetch_data['result'][pmid]
                        title = article.get('title', '')
                        source = article.get('source', '')
                        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        summary = f"{title} ({source}) [PubMed: {pubmed_url}]"
                        results.append(summary)
                logger.info("pubmed_search_complete", num_results=len(results))
                if self.cache:
                    self.cache.set(query, "pubmed", results)
                return results
        except Exception as e:
            logger.error("pubmed_search_error", error=str(e), query=query)
            return []

class DuckDuckGoSearch:
    BASE_URL = "https://api.duckduckgo.com/"

    def __init__(self, timeout: int = 5, cache: Optional[WebSearchCache] = None):
        self.timeout = timeout
        self.cache = cache

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
    async def _fetch_with_retry(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        async with session.get(url, timeout=self.timeout) as response:
            response.raise_for_status()
            return await response.json()

    async def search(self, query: str) -> List[str]:
        if self.cache:
            cached = self.cache.get(query, "duckduckgo")
            if cached is not None:
                return cached
        logger.info("duckduckgo_search_start", query=query)
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.BASE_URL}?q={query} medical&format=json&no_html=1"
                data = await self._fetch_with_retry(session, url)
                results = []
                if data.get('AbstractText'):
                    results.append(data['AbstractText'][:300])
                for topic in data.get('RelatedTopics', [])[:2]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append(topic['Text'][:200])
                logger.info("duckduckgo_search_complete", num_results=len(results))
                if self.cache:
                    self.cache.set(query, "duckduckgo", results)
                return results
        except Exception as e:
            logger.error("duckduckgo_search_error", error=str(e), query=query)
            return []
