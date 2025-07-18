// Admin panel JavaScript functionality
$(document).ready(function() {
    // Setup CSRF token for AJAX requests
    $.ajaxSetup({
        beforeSend: function(xhr, settings) {
            if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", $('meta[name="csrf-token"]').attr('content'));
            }
        }
    });
    
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
        $.post(`/admin/users/${token}/enable`, {})
        .done(function(data) {
            if (data.success) {
                location.reload();
            } else {
                alert('Error: ' + data.error);
            }
        })
        .fail(function() {
            alert('Error: Failed to enable user');
        });
    }
}

function disableUser(token) {
    const reason = prompt('Reason for disabling user:');
    if (reason) {
        $.post(`/admin/users/${token}/disable`, {
            reason: reason
        })
        .done(function(data) {
            if (data.success) {
                location.reload();
            } else {
                alert('Error: ' + data.error);
            }
        })
        .fail(function() {
            alert('Error: Failed to disable user');
        });
    }
}

function deleteUser(token) {
    if (confirm('Are you sure you want to delete this user? This action cannot be undone.')) {
        $.ajax({
            url: `/admin/users/${token}`,
            type: 'DELETE'
        })
        .done(function(data) {
            if (data.success) {
                location.reload();
            } else {
                alert('Error: ' + data.error);
            }
        })
        .fail(function() {
            alert('Error: Failed to delete user');
        });
    }
}

function rotateUserToken(token) {
    if (confirm('Are you sure you want to rotate this user\'s token? The old token will be invalidated.')) {
        $.post(`/admin/users/${token}/rotate`, {})
        .done(function(data) {
            if (data.success) {
                alert('Token rotated successfully. New token: ' + data.new_token);
                location.reload();
            } else {
                alert('Error: ' + data.error);
            }
        })
        .fail(function() {
            alert('Error: Failed to rotate token');
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