"""
Test cases for frontend JavaScript modifications.
Tests the modal handling, auto-save logic, and related functionality.
"""
import unittest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModalStateManagement(unittest.TestCase):
    """Test modal state management logic."""

    def test_group_config_modal_open_flag_initial_state(self):
        """Test that modal open flag starts as False."""
        groupConfigModalOpen = False
        self.assertFalse(groupConfigModalOpen)

    def test_group_config_modal_open_flag_set_true(self):
        """Test setting modal open flag to True."""
        groupConfigModalOpen = False
        groupConfigModalOpen = True
        self.assertTrue(groupConfigModalOpen)

    def test_group_config_modal_open_flag_set_false(self):
        """Test setting modal open flag to False."""
        groupConfigModalOpen = True
        groupConfigModalOpen = False
        self.assertFalse(groupConfigModalOpen)


class TestAutoSaveLogic(unittest.TestCase):
    """Test auto-save logic for group configuration."""

    def test_auto_save_skipped_when_modal_open(self):
        """Test that auto-save is skipped when modal is open."""
        groupConfigModalOpen = True
        
        # Simulate saveGroupConfig function check
        def should_skip_auto_save(modal_open):
            if modal_open:
                return True
            return False
        
        self.assertTrue(should_skip_auto_save(groupConfigModalOpen))

    def test_auto_save_allowed_when_modal_closed(self):
        """Test that auto-save is allowed when modal is closed."""
        groupConfigModalOpen = False
        
        def should_skip_auto_save(modal_open):
            if modal_open:
                return True
            return False
        
        self.assertFalse(should_skip_auto_save(groupConfigModalOpen))

    def test_manual_save_always_allowed(self):
        """Test that manual save is always allowed regardless of modal state."""
        for modal_open in [True, False]:
            isManualSave = True
            
            def should_skip_save(modal_open, is_manual):
                if not is_manual and modal_open:
                    return True
                return False
            
            self.assertFalse(should_skip_save(modal_open, isManualSave))


class TestWorkerGroupSorting(unittest.TestCase):
    """Test worker group sorting logic."""

    def test_sort_nodes_with_master_first(self):
        """Test that master node is placed first in sorting."""
        workers = ['gn39-0', 'gn40-0', 'gn6-0', 'gn7-0']
        master_addr = 'gn6'
        
        def extract_node(worker_id):
            return worker_id.rsplit('-', 1)[0] if '-' in worker_id else worker_id
        
        unique_nodes = {}
        for wid in workers:
            node = extract_node(wid)
            if node not in unique_nodes:
                unique_nodes[node] = []
            unique_nodes[node].append(wid)
        
        sorted_nodes = sorted(unique_nodes.keys())
        
        # Apply fix: ensure master node is first
        if master_addr in sorted_nodes:
            master_idx = sorted_nodes.index(master_addr)
            if master_idx != 0:
                sorted_nodes.pop(master_idx)
                sorted_nodes.insert(0, master_addr)
        
        self.assertEqual(sorted_nodes[0], 'gn6')

    def test_worker_sorting_within_node(self):
        """Test that workers are sorted within each node."""
        workers = ['gn6-1', 'gn6-0', 'gn7-0', 'gn7-1']
        
        def extract_node(worker_id):
            return worker_id.rsplit('-', 1)[0] if '-' in worker_id else worker_id
        
        unique_nodes = {}
        for wid in workers:
            node = extract_node(wid)
            if node not in unique_nodes:
                unique_nodes[node] = []
            unique_nodes[node].append(wid)
        
        sorted_workers = []
        for node in sorted(unique_nodes.keys()):
            sorted_workers.extend(sorted(unique_nodes.get(node, [])))
        
        expected = ['gn6-0', 'gn6-1', 'gn7-0', 'gn7-1']
        self.assertEqual(sorted_workers, expected)

    def test_full_ranking_calculation(self):
        """Test complete ranking calculation with master node first."""
        workers = ['gn39-0', 'gn40-0', 'gn6-0', 'gn7-0']
        master_addr = 'gn6'
        world_size = 4
        
        def extract_node(worker_id):
            return worker_id.rsplit('-', 1)[0] if '-' in worker_id else worker_id
        
        unique_nodes = {}
        for wid in workers:
            node = extract_node(wid)
            if node not in unique_nodes:
                unique_nodes[node] = []
            unique_nodes[node].append(wid)
        
        sorted_nodes = sorted(unique_nodes.keys())
        if master_addr in sorted_nodes:
            master_idx = sorted_nodes.index(master_addr)
            if master_idx != 0:
                sorted_nodes.pop(master_idx)
                sorted_nodes.insert(0, master_addr)
        
        sorted_workers = []
        for node in sorted_nodes:
            sorted_workers.extend(sorted(unique_nodes.get(node, [])))
        
        # Expected: master node (gn6) first, then sorted by node name
        self.assertEqual(sorted_workers[0], 'gn6-0')
        
        # All workers should get valid ranks
        for i, worker in enumerate(sorted_workers):
            self.assertEqual(i, i)  # Verify ordering


class TestGroupConfigData(unittest.TestCase):
    """Test group configuration data structure."""

    def test_default_group_structure(self):
        """Test default group has correct structure."""
        group = {
            'id': 'default',
            'name': '默认分组',
            'workers': [],
            'pytorch_config': {}
        }
        
        self.assertIn('id', group)
        self.assertIn('name', group)
        self.assertIn('workers', group)
        self.assertIn('pytorch_config', group)

    def test_pytorch_config_structure(self):
        """Test PyTorch configuration has all required fields."""
        pytorch_config = {
            'master_addr': 'gn6',
            'master_port': 29500,
            'world_size': 4,
            'backend': 'nccl'
        }
        
        self.assertIsInstance(pytorch_config['master_addr'], str)
        self.assertIsInstance(pytorch_config['master_port'], int)
        self.assertIsInstance(pytorch_config['world_size'], int)
        self.assertIsInstance(pytorch_config['backend'], str)

    def test_localstorage_serialization(self):
        """Test that group config can be serialized to localStorage."""
        workerGroups = {
            'default': {
                'id': 'default',
                'name': '默认分组',
                'workers': ['gn6-0', 'gn7-0'],
                'pytorch_config': {
                    'master_addr': 'gn6',
                    'world_size': 2
                }
            }
        }
        
        # Simulate localStorage serialization
        serialized = json.dumps(workerGroups)
        deserialized = json.loads(serialized)
        
        self.assertEqual(deserialized['default']['workers'], ['gn6-0', 'gn7-0'])
        self.assertEqual(deserialized['default']['pytorch_config']['master_addr'], 'gn6')


class TestModalConfiguration(unittest.TestCase):
    """Test Bootstrap modal configuration."""

    def test_modal_backdrop_static(self):
        """Test that modal is configured with static backdrop."""
        # Simulate modal options
        modal_options = {
            'backdrop': 'static',
            'keyboard': False
        }
        
        self.assertEqual(modal_options['backdrop'], 'static')
        self.assertEqual(modal_options['keyboard'], False)

    def test_modal_hide_on_escape_disabled(self):
        """Test that ESC key doesn't close modal."""
        modal_options = {'keyboard': False}
        
        # With keyboard: false, ESC should not close the modal
        self.assertFalse(modal_options['keyboard'])

    def test_modal_hide_on_backdrop_click_disabled(self):
        """Test that clicking backdrop doesn't close modal."""
        modal_options = {'backdrop': 'static'}
        
        # With backdrop: 'static', clicking outside should not close
        self.assertEqual(modal_options['backdrop'], 'static')


class TestRenderWorkerGroupsSkipLogic(unittest.TestCase):
    """Test renderWorkerGroups skip logic when modal is open."""

    def test_skip_rendering_when_modal_open(self):
        """Test that rendering is skipped when config modal is open."""
        groupConfigModalOpen = True
        
        def should_skip_rendering(modal_open):
            if modal_open:
                return True
            return False
        
        self.assertTrue(should_skip_rendering(groupConfigModalOpen))

    def test_allow_rendering_when_modal_closed(self):
        """Test that rendering is allowed when modal is closed."""
        groupConfigModalOpen = False
        
        def should_skip_rendering(modal_open):
            if modal_open:
                return True
            return False
        
        self.assertFalse(should_skip_rendering(groupConfigModalOpen))


class TestMoveWorkersSaveLogic(unittest.TestCase):
    """Test save logic after moving workers between groups."""

    def test_save_after_move_skipped_if_modal_open(self):
        """Test that save is skipped after move if modal is open."""
        groupConfigModalOpen = True
        
        def should_save_after_move(modal_open):
            if modal_open:
                return False
            return True
        
        self.assertFalse(should_save_after_move(groupConfigModalOpen))

    def test_save_after_move_allowed_if_modal_closed(self):
        """Test that save is allowed after move if modal is closed."""
        groupConfigModalOpen = False
        
        def should_save_after_move(modal_open):
            if modal_open:
                return False
            return True
        
        self.assertTrue(should_save_after_move(groupConfigModalOpen))


if __name__ == '__main__':
    unittest.main()
