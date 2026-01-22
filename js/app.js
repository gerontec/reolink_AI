// Globale Variablen
let selectedFaces = new Set();

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Checkbox Event Listeners
    document.querySelectorAll('.face-select').forEach(checkbox => {
        checkbox.addEventListener('change', updateSelection);
    });

    // Keyboard Shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeModal();
            closeDetailModal();
        }
    });
});

// Selection Management
function updateSelection() {
    selectedFaces.clear();
    document.querySelectorAll('.face-select:checked').forEach(checkbox => {
        selectedFaces.add(parseInt(checkbox.value));
    });
    
    document.getElementById('selectedCount').textContent = selectedFaces.size;
}

// Batch Rename
function showBatchRenameModal() {
    if (selectedFaces.size === 0) {
        alert('Bitte wählen Sie mindestens ein Gesicht aus.');
        return;
    }
    
    document.getElementById('renameModal').style.display = 'block';
    document.getElementById('newPersonName').focus();
}

async function executeBatchRename() {
    const newName = document.getElementById('newPersonName').value.trim();
    
    if (!newName) {
        alert('Bitte geben Sie einen Namen ein.');
        return;
    }
    
    if (!confirm(`${selectedFaces.size} Gesicht(er) als "${newName}" benennen?`)) {
        return;
    }
    
    try {
        const response = await fetch('api/rename_faces.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                face_ids: Array.from(selectedFaces),
                new_name: newName
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert(`✓ ${result.updated} Gesicht(er) erfolgreich umbenannt!`);
            location.reload();
        } else {
            alert('Fehler: ' + result.error);
        }
    } catch (error) {
        alert('Fehler beim Umbenennen: ' + error);
    }
}

// Single Rename
async function renameSingle(faceId) {
    const newName = prompt('Neuer Name:');
    
    if (!newName || !newName.trim()) {
        return;
    }
    
    try {
        const response = await fetch('api/rename_faces.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                face_ids: [faceId],
                new_name: newName.trim()
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            location.reload();
        } else {
            alert('Fehler: ' + result.error);
        }
    } catch (error) {
        alert('Fehler: ' + error);
    }
}

// Batch Delete
async function batchDelete() {
    if (selectedFaces.size === 0) {
        alert('Bitte wählen Sie mindestens ein Gesicht aus.');
        return;
    }
    
    if (!confirm(`${selectedFaces.size} Gesicht(er) wirklich löschen?`)) {
        return;
    }
    
    try {
        const response = await fetch('api/delete_faces.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                face_ids: Array.from(selectedFaces)
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert(`✓ ${result.deleted} Gesicht(er) gelöscht!`);
            location.reload();
        } else {
            alert('Fehler: ' + result.error);
        }
    } catch (error) {
        alert('Fehler: ' + error);
    }
}

// Single Delete
async function deleteSingle(faceId) {
    if (!confirm('Dieses Gesicht wirklich löschen?')) {
        return;
    }
    
    try {
        const response = await fetch('api/delete_faces.php', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                face_ids: [faceId]
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            location.reload();
        } else {
            alert('Fehler: ' + result.error);
        }
    } catch (error) {
        alert('Fehler: ' + error);
    }
}

// View Original Image
function viewOriginal(faceId) {
    showFaceDetail(faceId);
}

// Show Face Detail
async function showFaceDetail(faceId) {
    try {
        const response = await fetch(`api/face_detail.php?id=${faceId}`);
        const data = await response.json();
        
        if (data.success) {
            const detail = data.face;
            document.getElementById('detailContent').innerHTML = `
                <h2>Gesicht Details</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <img src="api/crop_face.php?id=${faceId}&size=400" 
                             style="width: 100%; border-radius: 8px;">
                    </div>
                    <div>
                        <h3>${detail.person_name}</h3>
                        <p><strong>Konfidenz:</strong> ${(detail.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Größe:</strong> ${detail.width}x${detail.height}px</p>
                        <p><strong>Kamera:</strong> ${detail.camera_name}</p>
                        <p><strong>Aufnahme:</strong> ${detail.recorded_at}</p>
                        <p><strong>Datei:</strong> ${detail.file_path}</p>
                        <br>
                        <a href="/web1/${detail.file_path}" target="_blank" class="btn btn-primary">
                            🖼️ Original-Bild öffnen
                        </a>
                    </div>
                </div>
            `;
            document.getElementById('detailModal').style.display = 'block';
        }
    } catch (error) {
        alert('Fehler beim Laden der Details: ' + error);
    }
}

// Modal Controls
function closeModal() {
    document.getElementById('renameModal').style.display = 'none';
    document.getElementById('newPersonName').value = '';
}

function closeDetailModal() {
    document.getElementById('detailModal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const renameModal = document.getElementById('renameModal');
    const detailModal = document.getElementById('detailModal');
    
    if (event.target === renameModal) {
        closeModal();
    }
    if (event.target === detailModal) {
        closeDetailModal();
    }
}
