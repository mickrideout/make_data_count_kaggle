#!/usr/bin/env python3
"""
Zotero Papers with Code Linker

Links papers in a Zotero collection with their corresponding GitHub repositories
from Papers with Code using fuzzy title matching.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

from pyzotero import zotero
from fuzzywuzzy import fuzz


class ZoteroPWCLinker:
    def __init__(self, user_id: str, api_key: str):
        self.zotero = zotero.Zotero(user_id, 'user', api_key)

    def get_collection_id(self, collection_name: str) -> Optional[str]:
        """Get collection ID by name"""
        collections = self.zotero.collections()
        
        for collection in collections:
            if collection['data']['name'] == collection_name:
                return collection['key']
        
        print(f"Collection '{collection_name}' not found")
        return None

    def get_collection_items(self, collection_id: str) -> List[Dict]:
        """Get all items in a collection"""
        return self.zotero.collection_items(collection_id)

    def find_best_match(self, paper_title: str, pwc_data: List[Dict]) -> Optional[Dict]:
        """Find the best matching paper using fuzzy string matching"""
        best_match = None
        best_ratio = 0
        threshold = 80  # Minimum similarity threshold
        
        for paper in pwc_data:
            pwc_title = paper.get('paper_title')
            if not pwc_title:
                continue
            ratio = fuzz.ratio(paper_title.lower(), pwc_title.lower())
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = paper
        
        return best_match

    def add_attachment_to_item(self, item_key: str, repo_url: str, paper_title: str) -> bool:
        """Add a web link attachment to a Zotero item"""
        attachment_data = {
            "itemType": "attachment",
            "linkMode": "linked_url",
            "title": f"CODE Repository - {paper_title}",
            "url": repo_url,
            "note": "Added by Zotero PWC Linker",
            "parentItem": item_key
        }
        
        try:
            self.zotero.create_items([attachment_data])
            print(f"✓ Added GitHub link for: {paper_title}")
            return True
        except Exception as e:
            print(f"✗ Failed to add attachment for: {paper_title} (Error: {e})")
            return False

    def has_code_repository_attachment(self, item_key: str) -> bool:
        """Check if an item already has a CODE Repository attachment"""
        try:
            children = self.zotero.children(item_key)
            for child in children:
                if (child['data']['itemType'] == 'attachment' and 
                    child['data'].get('title', '').startswith('CODE Repository')):
                    return True
            return False
        except Exception as e:
            print(f"Error fetching item children: {e}")
            return False

    def process_collection(self, collection_name: str, pwc_data: List[Dict]):
        """Process all papers in the collection and add GitHub links"""
        collection_id = self.get_collection_id(collection_name)
        if not collection_id:
            return
        
        items = self.get_collection_items(collection_id)
        processed = 0
        matched = 0
        skipped = 0
        
        for item in items:
            item_type = item['data'].get('itemType', '')
            if item_type not in ['journalArticle', 'preprint', 'conferencePaper', 'webpage', 'booksection']:
                continue
                
            title = item['data'].get('title', '')
            if not title:
                continue
                
            # Check if item already has a CODE Repository attachment
            if self.has_code_repository_attachment(item['key']):
                print(f"Skipping (already has CODE Repository): {title}")
                skipped += 1
                continue
                
            processed += 1
            match = self.find_best_match(title, pwc_data)
            
            if match:
                print(f"Found match: {title} -> {match['paper_title']}")
                if self.add_attachment_to_item(item['key'], match['repo_url'], title):
                    matched += 1
            else:
                print(f"No match found for: {title}")
        
        print(f"\nProcessing complete:")
        print(f"Papers processed: {processed}")
        print(f"Papers skipped (already have CODE Repository): {skipped}")
        print(f"GitHub links added: {matched}")


def main():
    parser = argparse.ArgumentParser(description='Link Zotero papers with GitHub repositories from Papers with Code')
    parser.add_argument('pwc_json_path', help='Path to the Papers with Code links JSON file')
    parser.add_argument('collection_name', help='Zotero collection name')
    parser.add_argument('--zotero-user-id', help='Zotero user ID', default="3841519")
    parser.add_argument('--zotero-api-key', help='Zotero API key', default="lMN3ZdjVGyZK3fYzQjOZJUlg")
    
    args = parser.parse_args()
    
    # Load Papers with Code data
    try:
        with open(args.pwc_json_path, 'r') as f:
            pwc_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file {args.pwc_json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.pwc_json_path}")
        sys.exit(1)
    
    # Create linker and process collection
    linker = ZoteroPWCLinker(args.zotero_user_id, args.zotero_api_key)
    linker.process_collection(args.collection_name, pwc_data)


if __name__ == "__main__":
    main()