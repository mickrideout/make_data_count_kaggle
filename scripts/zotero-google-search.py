#!/usr/bin/env python3
"""
Script to search for GitHub repositories related to papers in a Zotero collection.
Uses Google search to find code repositories and adds them to Zotero items.
"""

import argparse
import re
import time
from typing import List, Optional
from urllib.parse import urlparse

import requests
from pyzotero import zotero
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def has_code_repository(zot: zotero.Zotero, item_key: str) -> bool:
    """Check if a Zotero item already has a CODE Repository attachment."""
    try:
        children = zot.children(item_key)
        for child in children:
            if (child['data']['itemType'] == 'attachment' and 
                child['data'].get('title', '').startswith('CODE Repository')):
                return True
        return False
    except Exception as e:
        print(f"Error fetching item children: {e}")
        return False


def google_search(query: str, api_key: str, cse_id: str, num_results: int = 10) -> List[str]:
    """
    Perform Google Custom Search using the official Google API client library.
    
    Args:
        query: Search query
        api_key: Google API key
        cse_id: Custom Search Engine ID
        num_results: Number of results to return
    
    Returns:
        List of URLs from search results
    """
    try:
        # Build the service
        service = build("customsearch", "v1", developerKey=api_key)
        
        # Perform the search
        result = service.cse().list(
            q=query,
            cx=cse_id,
            num=min(num_results, 10)  # API limit is 10 per request
        ).execute()
        
        urls = []
        for item in result.get('items', []):
            urls.append(item['link'])
        
        return urls
        
    except HttpError as e:
        print(f"Error performing Google search: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []





def find_github_repos(urls: List[str]) -> List[str]:
    """Extract GitHub repository URLs from a list of URLs."""
    github_repos = []
    github_pattern = re.compile(r'https?://github\.com/[^/]+/[^/]+')
    
    for url in urls:
        if 'github.com' in url:
            match = github_pattern.match(url)
            if match:
                repo_url = match.group(0)
                if repo_url not in github_repos:
                    github_repos.append(repo_url)
    
    return github_repos


def add_code_repository_to_item(zot: zotero.Zotero, item_key: str, repo_url: str, paper_title: str = "") -> bool:
    """
    Add CODE Repository attachment to a Zotero item.
    
    Args:
        zot: Zotero client
        item_key: Zotero item key
        repo_url: GitHub repository URL
        paper_title: Title of the paper (for attachment title)
    
    Returns:
        True if successful, False otherwise
    """
    attachment_data = {
        "itemType": "attachment",
        "linkMode": "linked_url",
        "title": f"CODE Repository - {paper_title}" if paper_title else "CODE Repository",
        "url": repo_url,
        "note": "Added by Zotero Google Search",
        "parentItem": item_key
    }
    
    try:
        zot.create_items([attachment_data])
        print(f"✓ Added GitHub link for: {paper_title}")
        return True
    except Exception as e:
        print(f"✗ Failed to add attachment for: {paper_title} (Error: {e})")
        return False


def process_zotero_collection(user_id: str, api_key: str, collection_name: str, 
                            google_api_key: Optional[str] = None, 
                            google_cse_id: Optional[str] = None) -> None:
    """
    Process all papers in a Zotero collection to find and add GitHub repositories.
    
    Args:
        user_id: Zotero user ID
        api_key: Zotero API key
        collection_name: Name of the Zotero collection
        google_api_key: Google API key for Custom Search
        google_cse_id: Google Custom Search Engine ID
    """
    # Initialize Zotero client
    zot = zotero.Zotero(user_id, 'user', api_key)
    
    try:
        # Get all collections
        collections = zot.collections()
        target_collection = None
        
        for collection in collections:
            if collection['data']['name'] == collection_name:
                target_collection = collection
                break
        
        if not target_collection:
            print(f"Collection '{collection_name}' not found.")
            return
        
        collection_key = target_collection['key']
        print(f"Found collection: {collection_name}")
        
        # Get items in collection
        items = zot.collection_items(collection_key)
        print(f"Found {len(items)} items in collection")
        
        processed_count = 0
        skipped_count = 0
        updated_count = 0
        
        # Define allowed item types
        allowed_types = ["journalArticle", "preprint", "conferencePaper", "webpage", "booksection"]
        
        for item in items:
            item_data = item.get('data', {})
            title = item_data.get('title', '')
            item_key = item.get('key', '')
            item_type = item_data.get('itemType', '')
            
            if not title:
                print(f"Skipping item {item_key}: No title")
                skipped_count += 1
                continue
            
            # Check if item type is allowed
            if item_type not in allowed_types:
                #print(f"Skipping '{title}': Item type '{item_type}' not in allowed types")
                skipped_count += 1
                continue
            
            # Check if item already has CODE Repository
            if has_code_repository(zot, item_key):
                print(f"Skipping '{title}': Already has CODE Repository")
                skipped_count += 1
                continue
            
            print(f"Processing: {title}")
            
            # If Google API credentials are provided, perform search
            if google_api_key and google_cse_id:
                search_query = f'"{title}" implements'
                print(f"Searching for: {search_query}")
                
                # Perform Google search
                search_urls = google_search(search_query, google_api_key, google_cse_id)
                
                # Find GitHub repositories
                github_repos = find_github_repos(search_urls)
                
                if github_repos:
                    print(f"Found GitHub repositories: {github_repos}")
                    
                    # Add the first repository found
                    repo_url = github_repos[0]
                    if add_code_repository_to_item(zot, item_key, repo_url, title):
                        print(f"Added CODE Repository: {repo_url}")
                        updated_count += 1
                    else:
                        print(f"Failed to update item")
                else:
                    print("No GitHub repositories found")
                
                # Rate limiting - wait 1 second between requests
                time.sleep(1)
            else:
                print("Google API credentials not provided - skipping search")
            
            processed_count += 1
        
        print(f"\nProcessing complete:")
        print(f"Items processed: {processed_count}")
        print(f"Items skipped: {skipped_count}")
        print(f"Items updated: {updated_count}")
        
    except Exception as e:
        print(f"Error processing collection: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Search for GitHub repositories related to papers in a Zotero collection"
    )

    parser.add_argument("collection_name", help="Zotero collection name")
    parser.add_argument("--zotero-user-id", help="Zotero user ID", default="3841519")
    parser.add_argument("--zotero-api-key", help="Zotero API key", default="lMN3ZdjVGyZK3fYzQjOZJUlg")
    parser.add_argument("--google-api-key", help="Google API key for Custom Search (optional)", default="AIzaSyBbtjHfh4dTPFHTLNDlYE3AmTempDVkqYI")
    parser.add_argument("--google-cse-id", help="Google Custom Search Engine ID (optional)", default="a128db5d180a4494a")
    
    args = parser.parse_args()
    
    if args.google_api_key and not args.google_cse_id:
        print("Error: --google-cse-id is required when --google-api-key is provided")
        return
    
    if args.google_cse_id and not args.google_api_key:
        print("Error: --google-api-key is required when --google-cse-id is provided")
        return
    
    process_zotero_collection(
        args.zotero_user_id,
        args.zotero_api_key,
        args.collection_name,
        args.google_api_key,
        args.google_cse_id
    )


if __name__ == "__main__":
    main()