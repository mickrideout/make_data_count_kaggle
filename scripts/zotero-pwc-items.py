#!/usr/bin/env python3

# Given a Zotero collection name, find all items with a Papers with Code link.

import argparse
import sys
from typing import Dict, List, Any
from pyzotero import zotero


class ZoteroPwcItems:
    def __init__(self, user_id: str, collection_name: str, api_key: str):
        self.user_id = user_id
        self.collection_name = collection_name
        self.api_key = api_key
        
        if not api_key or len(api_key) < 20:
            raise ValueError("API key appears to be invalid. Zotero API keys are typically 32 characters long.")
        
        self.zot = zotero.Zotero(user_id, 'user', api_key)
        
    def get_collection_id(self) -> str:
        """Get the collection ID by name"""
        collections = self.zot.collections()
    
        
        for collection in collections:
            if collection['data']['name'] == self.collection_name:
                return collection['key']
        
        raise ValueError(f"Collection '{self.collection_name}' not found")
    
    def get_collection_items(self, collection_id: str) -> List[Dict[str, Any]]:
        """Get all items in the collection"""
        all_items = self.zot.collection_items(collection_id)
        return all_items
    
    def get_parent_item(self, item_key: str) -> Dict[str, Any]:
        """Get the parent item of a given item"""
        try:
            parent_item = self.zot.item(item_key)
            return parent_item
        except Exception as e:
            print(f"Error getting parent item for {item_key}: {e}")
            return None
    
    def process_collection(self):
        """Process all items in the collection and find those with PwC links"""
        try:
            collection_id = self.get_collection_id()
            items = self.get_collection_items(collection_id)
            
            print(f"Processing {len(items)} items in collection '{self.collection_name}'")
            print("-" * 50)
            
            pwc_items_found = 0
            
            for item in items:
                try:
                    if "data" not in item:
                        continue
                    
                    item_type = item["data"].get("itemType", "")
                    
                    if item_type == "attachment":
                        url = item["data"].get("url", "")
                        
                        if url and "https://paperswithcode.com/" in url:
                            parent_key = item["data"].get("parentItem")
                            
                            if parent_key:
                                parent_item = self.get_parent_item(parent_key)
                                
                                if parent_item and "data" in parent_item:
                                    title = parent_item["data"].get("title", "No title")
                                    creators = parent_item["data"].get("creators", [])
                                    
                                    
                                    authors = []
                                    for creator in creators:
                                        if creator.get("creatorType") == "author":
                                            name = creator.get("name", "")
                                            if name:
                                                authors.append(name)
                                    
                                    # Also check for alternative author fields
                                    if not authors:
                                        # Check for firstName/lastName structure
                                        for creator in creators:
                                            if creator.get("creatorType") == "author":
                                                first_name = creator.get("firstName", "")
                                                last_name = creator.get("lastName", "")
                                                if first_name or last_name:
                                                    full_name = f"{first_name} {last_name}".strip()
                                                    if full_name:
                                                        authors.append(full_name)
                                    
                                    author_str = ", ".join(authors) if authors else "No author"
                                    
                                    print(f"Title: {title}")
                                    print(f"Author: {author_str}")
                                    print(f"PwC URL: {url}")
                                    print("-" * 30)
                                    
                                    pwc_items_found += 1
                    
                except Exception as e:
                    print(f"Error processing item {item.get('key', 'unknown')}: {e}")
                    continue
            
            print(f"\nSummary: Found {pwc_items_found} items with Papers with Code links")
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Find Zotero items with Papers with Code links")
    parser.add_argument("user_id", help="Zotero user ID")
    parser.add_argument("api_key", help="Zotero API key")
    parser.add_argument("collection_name", help="Zotero collection name")
    
    args = parser.parse_args()
    
    print("Zotero PwC Items Finder")
    print("=" * 30)
    print()
    
    try:
        finder = ZoteroPwcItems(args.user_id, args.collection_name, args.api_key)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    finder.process_collection()
    
    print("Done!")


if __name__ == "__main__":
    main() 