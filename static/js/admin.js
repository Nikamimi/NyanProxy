// Admin panel JavaScript functionality
$(document).ready(function() {
    // Note: Using session-based authentication instead of CSRF tokens
    
    // Flash message auto-dismiss
    setTimeout(function() {
        $('.flash').fadeOut();
    }, 5000);
    
    // Confirm delete actions
    $('.delete-btn').on('click', function(e) {
        if (!confirm('Are you sure you want to delete this item?')) {
            e.preventDefault();
        }
    });
    
    // Auto-refresh dashboard stats
    if (window.location.pathname === '/admin' || window.location.pathname === '/admin/') {
        setInterval(refreshDashboardStats, 30000); // Refresh every 30 seconds
    }
});

function refreshDashboardStats() {
    $.get('/admin/api/stats', function(data) {
        if (data.total_users !== undefined) {
            $('.stat-number').eq(0).text(data.total_users);
            $('.stat-number').eq(1).text(data.active_users);
            $('.stat-number').eq(2).text(data.disabled_users);
            $('.stat-number').eq(3).text(data.usage_stats.total_requests || 0);
        }
    }).fail(function() {
        console.log('Failed to refresh dashboard stats');
    });
}

// User management functions
function enableUser(token) {
    if (confirm('Are you sure you want to enable this user?')) {
        $.ajax({
            url: `/admin/api/users/${token}/enable`,
            type: 'POST',
            contentType: 'application/json'
        })
        .done(function(data) {
            if (data.success) {
                console.log('User enabled:', data);
                // Update UI immediately instead of reloading
                updateUserStateInUI(token, data.user_state);
                showSuccessMessage(data.message);
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        })
        .fail(function(xhr) {
            console.error('Enable user error:', xhr);
            alert('Error: HTTP ' + xhr.status + ' - ' + xhr.statusText);
        });
    }
}

function disableUser(token) {
    const reason = prompt('Reason for disabling user:');
    if (reason) {
        $.ajax({
            url: `/admin/api/users/${token}/disable`,
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ reason: reason })
        })
        .done(function(data) {
            if (data.success) {
                console.log('User disabled:', data);
                // Update UI immediately instead of reloading
                updateUserStateInUI(token, data.user_state);
                showSuccessMessage(data.message);
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        })
        .fail(function(xhr) {
            console.error('Disable user error:', xhr);
            alert('Error: HTTP ' + xhr.status + ' - ' + xhr.statusText);
        });
    }
}

function deleteUser(token) {
    if (confirm('Are you sure you want to delete this user? This action cannot be undone.')) {
        $.ajax({
            url: `/admin/api/users/${token}`,
            type: 'DELETE',
            contentType: 'application/json'
        })
        .done(function(data) {
            if (data.success) {
                location.reload();
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        })
        .fail(function(xhr) {
            console.error('Delete user error:', xhr);
            alert('Error: HTTP ' + xhr.status + ' - ' + xhr.statusText);
        });
    }
}

function rotateUserToken(token) {
    if (confirm('Are you sure you want to rotate this user\'s token? The old token will be invalidated.')) {
        $.ajax({
            url: `/admin/api/users/${token}/rotate`,
            type: 'POST',
            contentType: 'application/json'
        })
        .done(function(data) {
            if (data.success) {
                alert('Token rotated successfully. New token: ' + data.new_token);
                location.reload();
            } else {
                alert('Error: ' + (data.error || 'Unknown error'));
            }
        })
        .fail(function(xhr) {
            console.error('Rotate token error:', xhr);
            alert('Error: HTTP ' + xhr.status + ' - ' + xhr.statusText);
        });
    }
}

// Bulk operations
function bulkRefreshQuotas() {
    if (confirm('Are you sure you want to refresh quotas for all users?')) {
        $.post('/admin/bulk/refresh-quotas', {})
        .done(function(data) {
            if (data.success) {
                alert(`Successfully refreshed quotas for ${data.affected_users} users`);
                location.reload();
            } else {
                alert('Error: ' + data.error);
            }
        })
        .fail(function() {
            alert('Error: Failed to refresh quotas');
        });
    }
}

// Copy to clipboard functionality
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        // Show temporary success message
        const msg = $('<div class="copy-success">Copied to clipboard!</div>');
        $('body').append(msg);
        setTimeout(function() {
            msg.fadeOut(function() { msg.remove(); });
        }, 2000);
    }).catch(function(err) {
        console.error('Failed to copy to clipboard:', err);
        alert('Failed to copy to clipboard');
    });
}

// Form validation
function validateUserForm() {
    const nickname = $('#nickname').val().trim();
    const openaiLimit = $('#openai_limit').val();
    const anthropicLimit = $('#anthropic_limit').val();
    
    if (!nickname) {
        alert('Please enter a nickname');
        return false;
    }
    
    if (openaiLimit && (isNaN(openaiLimit) || parseInt(openaiLimit) < 0)) {
        alert('OpenAI limit must be a positive number');
        return false;
    }
    
    if (anthropicLimit && (isNaN(anthropicLimit) || parseInt(anthropicLimit) < 0)) {
        alert('Anthropic limit must be a positive number');
        return false;
    }
    
    return true;
}

// Table sorting
function sortTable(columnIndex) {
    const table = document.querySelector('.users-table table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    rows.sort((a, b) => {
        const aText = a.cells[columnIndex].textContent.trim();
        const bText = b.cells[columnIndex].textContent.trim();
        
        // Try to parse as numbers first
        const aNum = parseFloat(aText);
        const bNum = parseFloat(bText);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return aNum - bNum;
        }
        
        // Otherwise, sort as strings
        return aText.localeCompare(bText);
    });
    
    tbody.innerHTML = '';
    rows.forEach(row => tbody.appendChild(row));
}

// Search functionality
function filterUsers() {
    const searchTerm = $('#user-search').val().toLowerCase();
    const rows = $('.users-table tbody tr');
    
    rows.each(function() {
        const row = $(this);
        const text = row.text().toLowerCase();
        
        if (text.includes(searchTerm)) {
            row.show();
        } else {
            row.hide();
        }
    });
}

// Real-time search
$('#user-search').on('input', function() {
    filterUsers();
});

// Export functionality
function exportUsers() {
    window.location.href = '/admin/export/users';
}

function exportEvents() {
    window.location.href = '/admin/export/events';
}

// UI update functions for real-time sync
function updateUserStateInUI(token, userState) {
    console.log('Updating UI for user:', token, 'State:', userState);
    
    // Find the user row in the table
    const userRow = $(`tr`).filter(function() {
        return $(this).find('code').text().startsWith(token.substring(0, 8));
    });
    
    if (userRow.length > 0) {
        // Update status column
        const statusCell = userRow.find('td').eq(4); // Status is the 5th column (0-indexed)
        if (userState.is_disabled) {
            statusCell.html('<span class="status-disabled">Disabled</span>');
            userRow.addClass('disabled');
        } else {
            statusCell.html('<span class="status-active">Active</span>');
            userRow.removeClass('disabled');
        }
        
        // Update action buttons
        const actionCell = userRow.find('td').eq(6); // Actions is the 7th column
        const actionButtons = actionCell.find('.action-buttons');
        
        // Remove existing enable/disable buttons
        actionButtons.find('.btn-success, .btn-warning').remove();
        
        // Add the appropriate button
        const editBtn = actionButtons.find('.btn-secondary');
        if (userState.is_disabled) {
            editBtn.after(`
                <button class="btn btn-sm btn-success" onclick="enableUser('${token}')">
                    <span class="cat-emoji">✅</span> Enable
                </button>
            `);
        } else {
            editBtn.after(`
                <button class="btn btn-sm btn-warning" onclick="disableUser('${token}')">
                    <span class="cat-emoji">⏸️</span> Disable
                </button>
            `);
        }
        
        console.log('UI updated for user:', token);
    } else {
        console.warn('Could not find user row for token:', token);
        // Fallback to page reload if we can't find the row
        setTimeout(() => location.reload(), 1000);
    }
}

function showSuccessMessage(message) {
    // Create a temporary success message
    const successDiv = $(`
        <div class="alert alert-success" style="position: fixed; top: 20px; right: 20px; z-index: 10000; padding: 15px; border-radius: 10px; background: rgba(76, 175, 80, 0.9); color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <span class="cat-emoji">✅</span> ${message}
        </div>
    `);
    
    $('body').append(successDiv);
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        successDiv.fadeOut(300, function() {
            $(this).remove();
        });
    }, 3000);
}