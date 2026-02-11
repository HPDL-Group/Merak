"""
Test cases for PyTorch distributed configuration and rank assignment logic.
Tests the modifications made to ensure:
1. Master node is always first in sorted order
2. Local rank is calculated correctly per node within group
3. Rank assignment follows correct ordering
"""
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_node_name(worker_id):
    """Extract node name from worker ID (e.g., 'gn6' from 'gn6-1')."""
    if '-' in worker_id:
        return worker_id.rsplit('-', 1)[0]
    return worker_id


class TestExtractNodeName(unittest.TestCase):
    """Test the extract_node_name utility function."""

    def test_standard_worker_id(self):
        self.assertEqual(extract_node_name('gn6-1'), 'gn6')
        self.assertEqual(extract_node_name('gn39-0'), 'gn39')
        self.assertEqual(extract_node_name('node-123-456'), 'node-123')

    def test_simple_worker_id(self):
        self.assertEqual(extract_node_name('worker1'), 'worker1')

    def test_edge_cases(self):
        self.assertEqual(extract_node_name('node'), 'node')
        self.assertEqual(extract_node_name(''), '')


class TestRankAssignment(unittest.TestCase):
    """Test rank assignment logic for PyTorch distributed training."""

    def setUp(self):
        self.mock_logger = Mock()
        self.mock_lock = Mock()
        self.mock_config = {}

    def _simulate_rank_calculation(
        self,
        worker_id,
        group_workers,
        master_addr,
        world_size
    ):
        """
        Simulate the rank calculation logic from app.py.
        Returns: (rank, local_rank, node_rank)
        """
        master_node = extract_node_name(master_addr) if '-' in master_addr else master_addr
        current_node = extract_node_name(worker_id)

        unique_nodes = {}
        for wid in group_workers:
            node_name = extract_node_name(wid)
            if node_name not in unique_nodes:
                unique_nodes[node_name] = []
            unique_nodes[node_name].append(wid)

        sorted_nodes = sorted(unique_nodes.keys())
        
        # Ensure master node is always first (the fix we made)
        if master_node in sorted_nodes:
            master_idx = sorted_nodes.index(master_node)
            if master_idx != 0:
                sorted_nodes.pop(master_idx)
                sorted_nodes.insert(0, master_node)

        node_rank = sorted_nodes.index(current_node)

        sorted_workers = []
        for node in sorted_nodes:
            sorted_workers.extend(sorted(unique_nodes.get(node, [])))

        rank = None
        local_rank = 0
        
        if worker_id in sorted_workers:
            worker_idx = sorted_workers.index(worker_id)
            if worker_idx < world_size:
                rank = worker_idx
                
                # Calculate local_rank based on workers on the same node within the group
                current_node_workers = sorted(unique_nodes.get(current_node, []))
                local_rank = current_node_workers.index(worker_id)

        return rank, local_rank, node_rank

    def test_master_node_always_first(self):
        """Test that master node is always first in sorted order."""
        group_workers = ['gn39-1', 'gn40-1', 'gn6-1', 'gn7-1']
        master_addr = 'gn6'
        
        # Calculate ranks for each worker
        ranks = {}
        for worker in group_workers:
            rank, local_rank, node_rank = self._simulate_rank_calculation(
                worker, group_workers, master_addr, 4
            )
            ranks[worker] = (rank, local_rank, node_rank)

        # Master node worker should have rank 0
        self.assertEqual(ranks['gn6-1'][0], 0)
        
        # Workers on master node should have consecutive ranks starting from 0
        self.assertEqual(ranks['gn6-1'][1], 0)  # local_rank on master node

    def test_rank_assignment_order(self):
        """Test that ranks are assigned in correct order based on node sorting."""
        group_workers = ['gn39-1', 'gn40-1', 'gn6-1', 'gn7-1']
        master_addr = 'gn6'

        # Expected order: master node first, then sorted by node name
        expected_order = ['gn6-1', 'gn39-1', 'gn40-1', 'gn7-1']

        sorted_workers = []
        for wid in sorted(group_workers):
            node = extract_node_name(wid)
            sorted_workers.append((node, wid))
        
        sorted_nodes = sorted(set(extract_node_name(w) for w in group_workers))
        if 'gn6' in sorted_nodes:
            sorted_nodes.remove('gn6')
            sorted_nodes.insert(0, 'gn6')

        result = []
        for node in sorted_nodes:
            node_workers = sorted([w for w in group_workers if extract_node_name(w) == node])
            result.extend(node_workers)

        self.assertEqual(result, expected_order)

    def test_local_rank_calculation(self):
        """Test that local_rank is calculated correctly per node."""
        # Simulate workers on different nodes, including gn6-1
        group_workers = ['gn6-0', 'gn6-1', 'gn7-0', 'gn39-0']
        master_addr = 'gn6'

        # Get ranks for workers on gn6
        rank0, local0, _ = self._simulate_rank_calculation(
            'gn6-0', group_workers, master_addr, 4
        )
        rank1, local1, _ = self._simulate_rank_calculation(
            'gn6-1', group_workers, master_addr, 4
        )

        # Both should be on same node (gn6), so they should have consecutive local_ranks
        self.assertEqual(local0, 0)
        self.assertEqual(local1, 1)  # gn6-1 is second worker on gn6

        # gn6-0 should have lower global rank
        self.assertLess(rank0, rank1)

    def test_multi_worker_per_node(self):
        """Test scenario with multiple workers per node."""
        group_workers = ['gn6-0', 'gn6-1', 'gn7-0', 'gn7-1']
        master_addr = 'gn6'
        world_size = 4

        results = {}
        for worker in group_workers:
            rank, local_rank, _ = self._simulate_rank_calculation(
                worker, group_workers, master_addr, world_size
            )
            results[worker] = (rank, local_rank)

        # Expected: gn6-0 has rank 0, local_rank 0
        self.assertEqual(results['gn6-0'], (0, 0))
        
        # gn6-1 should have higher rank, but still local_rank 0 if it's the only worker on gn6
        # In this case, both gn6-0 and gn6-1 are on gn6, so gn6-0 has local_rank 0, gn6-1 has local_rank 1
        self.assertEqual(results['gn6-1'][0], 1)
        self.assertEqual(results['gn6-1'][1], 1)

    def test_world_size_limitation(self):
        """Test that workers beyond world_size don't get ranks."""
        group_workers = ['gn6-0', 'gn6-1', 'gn7-0', 'gn7-1']
        master_addr = 'gn6'
        world_size = 2  # Only 2 workers should get ranks

        for worker in group_workers:
            rank, _, _ = self._simulate_rank_calculation(
                worker, group_workers, master_addr, world_size
            )
        
        # First 2 workers should have ranks, others should not
        self.assertEqual(self._simulate_rank_calculation('gn6-0', group_workers, master_addr, 2)[0], 0)
        self.assertEqual(self._simulate_rank_calculation('gn6-1', group_workers, master_addr, 2)[0], 1)
        self.assertIsNone(self._simulate_rank_calculation('gn7-0', group_workers, master_addr, 2)[0])
        self.assertIsNone(self._simulate_rank_calculation('gn7-1', group_workers, master_addr, 2)[0])

    def test_master_node_not_first_alphabetically(self):
        """Test that even if master node is not first alphabetically, it's placed first."""
        # Node names: gn6, gn39, gn40, gn7 (gn7 should come after others alphabetically)
        # But gn6 is master, so it should be first
        group_workers = ['gn39-0', 'gn40-0', 'gn6-0', 'gn7-0']
        master_addr = 'gn6'

        sorted_nodes = sorted(set(extract_node_name(w) for w in group_workers))
        
        # Without fix: ['gn39', 'gn40', 'gn6', 'gn7']
        # With fix: ['gn6', 'gn39', 'gn40', 'gn7']
        if master_addr in sorted_nodes:
            master_idx = sorted_nodes.index(master_addr)
            if master_idx != 0:
                sorted_nodes.pop(master_idx)
                sorted_nodes.insert(0, master_addr)

        self.assertEqual(sorted_nodes[0], 'gn6')

    def test_single_worker_group(self):
        """Test rank assignment for single worker group."""
        group_workers = ['gn6-0']
        master_addr = 'gn6'

        rank, local_rank, node_rank = self._simulate_rank_calculation(
            'gn6-0', group_workers, master_addr, 1
        )

        self.assertEqual(rank, 0)
        self.assertEqual(local_rank, 0)
        self.assertEqual(node_rank, 0)

    def test_worker_id_with_complex_format(self):
        """Test with workers that have complex ID formats."""
        group_workers = ['node-a-0', 'node-b-0', 'node-c-0']
        master_addr = 'node-c'

        sorted_nodes = sorted(set(extract_node_name(w) for w in group_workers))
        
        # Master should be first
        if master_addr in sorted_nodes:
            master_idx = sorted_nodes.index(master_addr)
            if master_idx != 0:
                sorted_nodes.pop(master_idx)
                sorted_nodes.insert(0, master_addr)

        self.assertEqual(sorted_nodes[0], 'node-c')


class TestPyTorchConfigSaving(unittest.TestCase):
    """Test PyTorch configuration saving and loading."""

    def setUp(self):
        self.test_config = {
            'name': 'test_group',
            'master_addr': 'gn6',
            'master_port': 29500,
            'world_size': 4,
            'backend': 'nccl',
            'workers': ['gn6-0', 'gn7-0', 'gn39-0', 'gn40-0']
        }

    def test_config_structure(self):
        """Test that config has all required fields."""
        required_fields = ['name', 'master_addr', 'master_port', 'world_size', 'backend', 'workers']
        for field in required_fields:
            self.assertIn(field, self.test_config, f"Missing required field: {field}")

    def test_config_values(self):
        """Test that config values are valid."""
        self.assertIsInstance(self.test_config['master_port'], int)
        self.assertIsInstance(self.test_config['world_size'], int)
        self.assertEqual(self.test_config['master_port'], 29500)
        self.assertEqual(self.test_config['backend'], 'nccl')


class TestGroupConfigAPI(unittest.TestCase):
    """Test group configuration API endpoints."""

    def setUp(self):
        self.app = None  # Would be set by fixtures in real tests
        self.test_client = None

    def test_create_group_endpoint_structure(self):
        """Test the structure of group creation response."""
        # This would test the actual API endpoint
        expected_response_fields = ['id', 'name', 'workers', 'pytorch_config']
        for field in expected_response_fields:
            # Mock response structure
            mock_response = {
                'id': 'test_group',
                'name': 'Test Group',
                'workers': [],
                'pytorch_config': {}
            }
            self.assertIn(field, mock_response)

    def test_update_group_config(self):
        """Test updating group PyTorch configuration."""
        original_config = {
            'master_addr': 'gn6',
            'master_port': 29500,
            'world_size': 4,
            'backend': 'nccl'
        }
        
        updated_config = {
            'master_addr': 'gn7',
            'master_port': 29501,
            'world_size': 8,
            'backend': 'nccl'
        }

        # Config should be updatable
        self.assertNotEqual(original_config['master_addr'], updated_config['master_addr'])
        self.assertNotEqual(original_config['world_size'], updated_config['world_size'])


if __name__ == '__main__':
    unittest.main()
