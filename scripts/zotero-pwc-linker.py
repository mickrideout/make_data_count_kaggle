#!/usr/bin/env python3

# Given a Zotero collection name, find all items with a Papers with Code link.

# Download the "All papers with abstracts" from here: https://github.com/paperswithcode/paperswithcode-data and decompress it.

# Pip install these:
# pyzotero
# fuzzywuzzy
# python-Levenshtein


import argparse
import json
import sys
from typing import Dict, List, Any
from pyzotero import zotero
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


class ZoteroPwcLinker:
    def __init__(self, api_key: str, user_id: str, collection_name: str):
        self.api_key = api_key
        self.user_id = user_id
        self.collection_name = collection_name
        
        # Validate API key format
        if not api_key or len(api_key) < 20:
            raise ValueError("API key appears to be invalid. Zotero API keys are typically 32 characters long.")
        
        # Initialise pyzotero client
        self.zot = zotero.Zotero(user_id, 'user', api_key)
        # Test the connection
        try:
            print("Testing Zotero connection...")
            test_collections = self.zot.collections()
            print(f"Successfully connected to Zotero. Found {len(test_collections)} collections.")
        except Exception as e:
            print(f"Failed to connect to Zotero: {e}")
            raise
        
    def get_collection_id(self) -> str:
        """Get the collection ID by name"""
        collections = self.zot.collections()
        
        for collection in collections:
            if collection['data']['name'] == self.collection_name:
                return collection['key']
        
        raise ValueError(f"Collection '{self.collection_name}' not found")
    
    def get_collection_items(self, collection_id: str) -> List[Dict[str, Any]]:
        """Get all items in the collection"""
        print(f"Fetching items from collection {collection_id}...")
        all_items = self.zot.collection_items(collection_id)
        print(f"Retrieved {len(all_items)} total items")
        # Filter out attachments and only return top-level items
        top_items = [item for item in all_items if item['data'].get('parentItem') is None]
        print(f"Found {len(top_items)} top-level items")
        return top_items
    
    def find_best_match(self, paper_title: str, pwc_data: List[Dict[str, Any]], threshold: int = 80) -> Dict[str, Any]:
        """Find the best matching paper title using fuzzy matching"""
        titles = [item["title"] for item in pwc_data]
        best_match = process.extractOne(paper_title, titles, scorer=fuzz.token_sort_ratio)
        
        if best_match and best_match[1] >= threshold:
            match_title = best_match[0]
            return next(item for item in pwc_data if item["title"] == match_title)
        return None
    
    def add_web_link_attachment(self, item_key: str, url: str, title: str = "Papers with Code"):
        """Add a web link attachment to a Zotero item"""
        attachment_data = {
            "itemType": "attachment",
            "linkMode": "linked_url",
            "url": url,
            "title": title,
            "parentItem": item_key,
            "accessDate": ""
        }
        
        try:
            # Create attachment using pyzotero
            self.zot.create_items([attachment_data])
            return True
        except Exception as e:
            raise Exception(f"Failed to create attachment: {e}")
    
    def process_collection(self, pwc_data: List[Dict[str, Any]]):
        """Process all items in the collection and add PwC links where matches are found"""
        try:
            print("Getting collection ID...")
            collection_id = self.get_collection_id()
            print(f"Collection ID: {collection_id}")
            items = self.get_collection_items(collection_id)
            print("Saving debug data...")
            with open("/tmp/debug.json", "w", encoding="utf-8") as debug_file:
                json.dump(items, debug_file, ensure_ascii=False, indent=2)
            
            print(f"Processing {len(items)} items in collection '{self.collection_name}'")
            
            matches_found = 0
            skipped_items = 0
            for item in items:
                try:
                    if "data" not in item:
                        print(f"- Skipping item without data: {item.get('key', 'unknown')}")
                        skipped_items += 1
                        continue
                        
                    item_type = item["data"].get("itemType", "")
                    title = item["data"].get("title", "")
                    
                    if item_type in ["journalArticle", "conferencePaper", "preprint", "webpage"]:
                        if title:
                            match = self.find_best_match(title, pwc_data)
                            if match:
                                try:
                                    existing_attachments = self.zot.children(item["key"])
                                    already_linked = False
                                    for att in existing_attachments:
                                        if att["data"].get("itemType") == "attachment" and att["data"].get("linkMode") == "linked_url":
                                            url = att["data"].get("url", "")
                                            if url and url.strip().startswith("https://paperswithcode.com/"):
                                                already_linked = True
                                                break
                                    if already_linked:
                                        print(f"- PwC link already exists for: {title}")
                                    else:
                                        self.add_web_link_attachment(
                                            item["key"], 
                                            match["paper_url"],
                                            f"Papers with Code: {match['title']}"
                                        )
                                        print(f"✓ Added PwC link to: {title}")
                                        matches_found += 1
                                except Exception as e:
                                    print(f"✗ Failed to add attachment for '{title}': {e}")
                                    print(f"  Continuing with next paper...")
                            else:
                                print(f"- No match found for: {title}")
                        else:
                            print(f"- Skipping {item_type} with no title")
                    else:
                        print(f"- Skipping {item_type} item")
                        skipped_items += 1
                except Exception as e:
                    print(f"✗ Error processing item: {e}")
                    print(f"  Item key: {item.get('key', 'unknown')}")
                    print(f"  Continuing with next paper...")
                    continue
            
            print(f"\nSummary: {matches_found} matches found and linked, {skipped_items} items skipped")
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def load_pwc_data(file_path: str) -> List[Dict[str, Any]]:
    """Load Papers with Code data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading PwC data: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Link Zotero papers with Papers with Code")
    parser.add_argument("pwc_file", help="Path to the papers-with-code links JSON file")
    parser.add_argument("user_id", help="Zotero user ID")
    parser.add_argument("api_key", help="Zotero API key")
    parser.add_argument("collection_name", help="Zotero collection name")
    parser.add_argument("--threshold", type=int, default=80, 
                       help="Fuzzy matching threshold (default: 80)")
    
    args = parser.parse_args()
    
    print("Zotero API Key Setup:")
    print("If you don't have an API key, get one at: https://www.zotero.org/settings/keys")
    print("Make sure to enable 'Allow library access' for the key.")
    print()
    
    print("Loading Papers with Code data...")
    pwc_data = load_pwc_data(args.pwc_file)
    print(f"Loaded {len(pwc_data)} papers from PwC data")
    
    print("Initialising Zotero linker...")
    try:
        linker = ZoteroPwcLinker(args.api_key, args.user_id, args.collection_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("Processing collection...")
    linker.process_collection(pwc_data)
    
    print("Done!")


if __name__ == "__main__":
    main()
