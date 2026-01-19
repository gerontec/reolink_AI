#!/home/gh/python/venv_py311/bin/python3
"""
Face Cluster Management Tool
Manage unknown face clusters detected by the system
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, '/home/gh/python')

from face_handler import get_handler, list_clusters, merge_clusters, rename_cluster

def show_help():
    print("""
╔════════════════════════════════════════════════════════════════╗
║              Face Cluster Management Tool                      ║
╚════════════════════════════════════════════════════════════════╝

Commands:

  list                    - List all unknown face clusters
  info <cluster_id>       - Show details about a cluster
  merge <id1> <id2>       - Merge two clusters (same person)
  rename <id> <name>      - Identify and rename a cluster
  delete <cluster_id>     - Delete a cluster
  stats                   - Show cluster statistics
  export                  - Export clusters to JSON
  
Examples:

  # List all clusters
  python3 face_clusters.py list
  
  # Merge Unknown_3 into Unknown_1 (same person)
  python3 face_clusters.py merge Unknown_1 Unknown_3
  
  # Identify Unknown_1 as "Nachbar"
  python3 face_clusters.py rename Unknown_1 Nachbar
  
  # Show statistics
  python3 face_clusters.py stats
""")

def cmd_list():
    """List all clusters"""
    clusters = list_clusters()
    
    if not clusters:
        print("📭 No unknown face clusters yet")
        return
    
    print(f"\n👥 Unknown Face Clusters: {len(clusters)}\n")
    print("ID                    Embeddings")
    print("─" * 40)
    
    for cluster_id, count in sorted(clusters.items()):
        print(f"{cluster_id:20} {count:>3} samples")
    
    print()

def cmd_stats():
    """Show cluster statistics"""
    handler = get_handler()
    clusters = handler.unknown_faces
    
    if not clusters:
        print("📭 No unknown face clusters yet")
        return
    
    total_embeddings = sum(len(embs) for embs in clusters.values())
    avg_embeddings = total_embeddings / len(clusters)
    
    print(f"\n📊 Cluster Statistics\n")
    print(f"Total Clusters:      {len(clusters)}")
    print(f"Total Embeddings:    {total_embeddings}")
    print(f"Avg per Cluster:     {avg_embeddings:.1f}")
    print(f"Next Cluster ID:     {handler.next_cluster_id}")
    print(f"Cluster Threshold:   {handler.cluster_threshold}")
    print()

def cmd_merge(cluster1, cluster2):
    """Merge two clusters"""
    print(f"🔄 Merging {cluster2} into {cluster1}...")
    
    if merge_clusters(cluster1, cluster2):
        print(f"✅ Successfully merged!")
        cmd_list()
    else:
        print(f"❌ Merge failed - clusters not found")

def cmd_rename(cluster_id, new_name):
    """Rename a cluster"""
    print(f"✏️  Renaming {cluster_id} to {new_name}...")
    
    if rename_cluster(cluster_id, new_name):
        print(f"✅ Successfully renamed!")
        cmd_list()
    else:
        print(f"❌ Rename failed - cluster not found")

def cmd_delete(cluster_id):
    """Delete a cluster"""
    handler = get_handler()
    
    if cluster_id not in handler.unknown_faces:
        print(f"❌ Cluster {cluster_id} not found")
        return
    
    count = len(handler.unknown_faces[cluster_id])
    
    print(f"⚠️  Delete {cluster_id} ({count} embeddings)?")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() == 'yes':
        del handler.unknown_faces[cluster_id]
        handler._save_unknown_faces()
        print(f"✅ Cluster deleted")
        cmd_list()
    else:
        print("❌ Cancelled")

def cmd_info(cluster_id):
    """Show cluster details"""
    handler = get_handler()
    
    if cluster_id not in handler.unknown_faces:
        print(f"❌ Cluster {cluster_id} not found")
        return
    
    embeddings = handler.unknown_faces[cluster_id]
    
    print(f"\n👤 Cluster: {cluster_id}\n")
    print(f"Embeddings:  {len(embeddings)}")
    print(f"Max Storage: 5 per cluster")
    print()
    
    # Get recent detections from DB
    try:
        import pymysql
        conn = pymysql.connect(
            host='192.168.178.218',
            user='gh',
            password='a12345',
            database='wagodb'
        )
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT timestamp, distance, video_id
                FROM cam_face_recognitions
                WHERE person_name = %s
                ORDER BY timestamp DESC
                LIMIT 10
            """, (cluster_id,))
            
            results = cursor.fetchall()
            
            if results:
                print(f"Recent Detections:")
                print("─" * 60)
                for ts, dist, vid in results:
                    print(f"{ts}  d={dist:.3f}  video={vid or 'N/A'}")
            else:
                print("No detections in database yet")
        
        conn.close()
    except Exception as e:
        print(f"⚠️  Could not fetch DB data: {e}")
    
    print()

def cmd_export():
    """Export clusters to JSON"""
    import json
    from datetime import datetime
    
    handler = get_handler()
    
    # Convert to serializable format
    export_data = {
        'exported_at': datetime.now().isoformat(),
        'clusters': {},
        'next_id': handler.next_cluster_id,
        'threshold': handler.cluster_threshold
    }
    
    for cluster_id, embeddings in handler.unknown_faces.items():
        export_data['clusters'][cluster_id] = {
            'count': len(embeddings),
            'embeddings': [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
        }
    
    filename = f"face_clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✅ Exported to {filename}")
    print(f"   Clusters: {len(export_data['clusters'])}")
    print(f"   Size: {os.path.getsize(filename) / 1024:.1f} KB")

def main():
    if len(sys.argv) < 2:
        show_help()
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'help' or cmd == '--help' or cmd == '-h':
        show_help()
    
    elif cmd == 'list':
        cmd_list()
    
    elif cmd == 'stats':
        cmd_stats()
    
    elif cmd == 'merge' and len(sys.argv) >= 4:
        cmd_merge(sys.argv[2], sys.argv[3])
    
    elif cmd == 'rename' and len(sys.argv) >= 4:
        cmd_rename(sys.argv[2], sys.argv[3])
    
    elif cmd == 'delete' and len(sys.argv) >= 3:
        cmd_delete(sys.argv[2])
    
    elif cmd == 'info' and len(sys.argv) >= 3:
        cmd_info(sys.argv[2])
    
    elif cmd == 'export':
        cmd_export()
    
    else:
        print(f"❌ Unknown command: {cmd}")
        print("Run 'python3 face_clusters.py help' for usage")

if __name__ == "__main__":
    main()
