from hashlib import blake2s, sha256

import torch
import torch.distributed as dist

from Merak import get_logger


class NVMapping(object):
    """
    A utility class that tracks and retrieves named parameters and optimizers through hash value mappings.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = get_logger("simple")
        self.parameter_name_hashval_dict = {}
        self.optimizer_name_hashval_dict = {}
        self.parameter_hashval_counter = {}
        self.optimizer_hashval_counter = {}

    def empty_cache(self):
        """
        Initialize the HashTracker with empty dictionaries for storing hash values.
        """
        self.parameter_name_hashval_dict = {}
        self.optimizer_name_hashval_dict = {}
        self.parameter_hashval_counter = {}
        self.optimizer_hashval_counter = {}

    def get_hash_value(self, original):
        """Generate a cryptographic hash value from a PyTorch tensor.

        Converts the input tensor to a CPU-based numpy array, then computes a BLAKE2s
        hash with 'optimizer' string as salt for additional uniqueness. The resulting
        hash is compact (7-byte digest) and suitable for checksum or identification purposes.

        Args:
            original (torch.Tensor): Input tensor (can be on GPU or CPU).
                Must be convertible to numpy array.

        Returns:
            str: 14-character hexadecimal string representation of the hash digest.

        Note:
            - Automatically handles device transfer (GPU→CPU) if needed
            - Uses deterministic salt ('optimizer') to prevent hash collisions
            - The 7-byte digest provides 56-bit security against collisions
        """
        val_cpu = original.cpu()
        array = val_cpu.detach().numpy()
        hash_value = blake2s(
            array.tobytes() + "optimizer".encode(), digest_size=7
        ).hexdigest()
        return hash_value

    # For parameters
    def initialization(self, value, name, is_zero3=False):
        """Initialize tracking for a parameter tensor by generating and storing its hash value.

        Handles both regular PyTorch tensors and DeepSpeed Zero3 partitioned tensors by:
        - Generating a unique hash for the tensor value
        - Storing the hash in two tracking dictionaries:
            1. Maps parameter names to their hash values
            2. Counts occurrences of each hash value (initialized to -1)

        Args:
            value: The parameter tensor to track.
                For Zero3 tensors, accesses the base tensor via .ds_tensor attribute.
            name (str): The parameter name used as dictionary key.
            is_zero3 (bool, optional): Whether the input is a DeepSpeed Zero3 tensor.
                Defaults to False.

        Side Effects:
            - Updates self.parameter_name_hashval_dict with {name: hash}
            - Initializes self.parameter_hashval_counter[hash] to -1

        Note:
            - The -1 initialization indicates the parameter hasn't been processed yet
            - Zero3 support allows compatibility with DeepSpeed's memory optimization
        """
        if not is_zero3:
            h_val = self.get_hash_value(value)
        else:
            h_val = self.get_hash_value(value.ds_tensor)

        self.parameter_name_hashval_dict[name] = h_val
        self.parameter_hashval_counter[h_val] = -1

    def get_name_from_parameters(self, value):
        """Retrieve the parameter name associated with a given tensor value using hash matching.

        Searches through the stored parameter dictionary to find the name corresponding to the
        input tensor's hash value. Maintains and updates collision counters for hash values that
        appear multiple times in the parameter set.

        Args:
            value (torch.Tensor): The tensor value to look up in the parameter dictionary.

        Returns:
            str: The name of the matching parameter if found, None otherwise.

        Note:
            - Uses distributed rank for logging warnings when parameter not found
            - Handles hash collisions by tracking occurrence counts
            - Updates collision counters when returning a matched parameter
            - The last two lines appear to be unreachable code (after return None)

        Warning:
            Logs a distributed warning message if the parameter cannot be found.
        """
        my_rank = dist.get_rank()
        val_count = 0
        h_val = self.get_hash_value(value)
        for name, val in self.parameter_name_hashval_dict.items():
            if h_val == val:
                if val_count > self.parameter_hashval_counter[h_val]:
                    self.parameter_hashval_counter[h_val] = val_count
                    self.logger.info(f'Find name: {name} for hashval:{h_val} with val:{value}')
                    return name
                val_count += 1
        self.logger.warning(
            f"Not find name for {value} from parameter dict", ranks=[my_rank]
        )
        return None

    # For optimizer
    def optimizer_param_name_mapping(self, value, name):
        """Maps optimizer parameter names to their hash values and initializes tracking counters.

        Stores the relationship between parameter names and their hash values in a dictionary,
        and initializes a counter for each hash value to track usage. Logs duplicate entries
        if the parameter name already exists in the mapping dictionary.

        Args:
            value (torch.Tensor): The parameter tensor to be hashed and tracked.
            name (str): The name of the parameter being mapped.

        Side Effects:
            - Updates self.optimizer_name_hashval_dict with {name: hash_value} mapping
            - Initializes self.optimizer_hashval_counter[hash_value] to -1
            - Logs informational message if parameter name already exists

        Note:
            - The -1 initialization indicates the parameter hasn't been processed yet
            - Uses distributed logging to maintain consistency across ranks
            - Hash values are generated using the class's get_hash_value method
        """
        my_rank = dist.get_rank()
        if name in self.optimizer_name_hashval_dict:
            self.logger.info(
                f"{name} already in optimizer_name_hashval_dict", ranks=[my_rank]
            )
        h_val = self.get_hash_value(value)
        self.optimizer_name_hashval_dict[name] = h_val
        self.optimizer_hashval_counter[h_val] = -1

    def get_name_from_optimizer(self, value):
        """Look up and return the parameter name associated with a given optimizer tensor value.

        Searches through the optimizer's name-to-hash dictionary to find the parameter name
        corresponding to the input tensor's hash value. Handles hash collisions by tracking
        and updating occurrence counts. Logs distributed warnings if the parameter isn't found.

        Args:
            value (torch.Tensor): The optimizer tensor value to look up in the tracking system.

        Returns:
            str: The name of the matching parameter if found, None otherwise.

        Side Effects:
            - Updates self.optimizer_hashval_counter with latest collision count
            - Logs distributed warning message if parameter not found

        Note:
            - Uses hash value comparison for efficient lookup
            - Maintains distributed logging consistency across ranks
            - The val_count > counter check ensures most recent collision is returned
            - Returns None (rather than raising exception) for missing parameters
        """
        val_count = 0
        h_val = self.get_hash_value(value)
        for name, val in self.optimizer_name_hashval_dict.items():
            if h_val == val:
                if val_count > self.optimizer_hashval_counter[h_val]:
                    self.optimizer_hashval_counter[h_val] = val_count
                    return name
                val_count += 1
        my_rank = dist.get_rank()
        self.logger.warning(
            f"Not find name for {value} from optimizer dict", ranks=[my_rank]
        )
        return None

    def find_name_with_equal_hash_value(self):
        """Detect and report hash collisions in both parameter and optimizer dictionaries.

        Scans through the stored parameter and optimizer name-to-hash mappings to:
        1. Identify pairs of names with identical hash values
        2. Log collision information across all ranks
        3. Provide dictionary size statistics for debugging

        Side Effects:
            - Logs collision pairs with format: 'In the [dict], name1 &&& name2 has
              equal hash value'
            - Reports dictionary sizes for both parameter and optimizer mappings
            - Outputs complete dictionary contents for verification

        Note:
            - Uses distributed logging to ensure consistency across ranks
            - Explicitly checks for name inequality (k1 != k2) to avoid false positives
            - Provides comprehensive debugging information for hash collision analysis
        """
        my_rank = dist.get_rank()
        # my_rank = 3  # Debug line - should be removed in production

        # Check parameter dictionary
        self.logger.info(
            f"Rank:{my_rank} len(self.parameter_name_hashval_dict.keys()):{len(self.parameter_name_hashval_dict.keys())}, "
            f"len(self.optimizer_name_hashval_dict.keys()):{len(self.optimizer_name_hashval_dict.keys())}",
            ranks=[my_rank],
        )
        self.logger.info(
            f"Rank:{my_rank}, self.parameter_name_hashval_dict.keys: {self.parameter_name_hashval_dict.keys()}",
            ranks=[my_rank],
        )
        for k1, v1 in self.parameter_name_hashval_dict.items():
            for k2, v2 in self.parameter_name_hashval_dict.items():
                if v1 == v2 and k1 != k2:
                    self.logger.info(
                        f"Rank:{my_rank}, in the parameter dict, {k1} &&& {k2} has equal hash value.",
                        ranks=[my_rank],
                    )

        # Check optimizer dictionary
        self.logger.info(
            f"Rank:{my_rank}, self.optimizer_name_hashval_dict.keys(): {self.optimizer_name_hashval_dict.keys()}",
            ranks=[my_rank],
        )
        for k1, v1 in self.optimizer_name_hashval_dict.items():
            for k2, v2 in self.optimizer_name_hashval_dict.items():
                if v1 == v2 and k1 != k2:
                    self.logger.info(
                        f"Rank:{my_rank}, in the optimizer dict, {k1} &&& {k2} has equal hash value.",
                        ranks=[my_rank],
                    )
