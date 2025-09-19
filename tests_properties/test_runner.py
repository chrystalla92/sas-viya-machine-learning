"""
Test runner for SAS Autoencoder Properties tests.

This script runs all property tests and provides a comprehensive report
on which SAS autoencoder properties are correctly implemented.
"""

import unittest
import sys
import time
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests_properties.test_architecture import TestArchitectureProperties
from tests_properties.test_preprocessing import TestPreprocessingProperties
from tests_properties.test_training import TestTrainingProperties
from tests_properties.test_output_transforms import TestOutputTransformProperties


class PropertyTestResult:
    """Container for test results with property categorization."""

    def __init__(self):
        self.architecture_results = {}
        self.preprocessing_results = {}
        self.training_results = {}
        self.output_transform_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.execution_time = 0.0

    def add_result(self, category: str, test_name: str, result: str, details: str = ""):
        """Add a test result to the appropriate category."""
        category_dict = getattr(self, f"{category}_results", {})
        category_dict[test_name] = {
            'result': result,
            'details': details
        }

        self.total_tests += 1
        if result == 'PASS':
            self.passed_tests += 1
        elif result == 'FAIL':
            self.failed_tests += 1
        else:  # ERROR
            self.error_tests += 1

    def print_summary(self):
        """Print a comprehensive test summary."""
        print("=" * 80)
        print("SAS AUTOENCODER PROPERTIES TEST SUMMARY")
        print("=" * 80)
        print(f"Execution Time: {self.execution_time:.2f} seconds")
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Errors: {self.error_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        print()

        # Print detailed results by category
        self._print_category_results("Architecture Properties", self.architecture_results)
        self._print_category_results("Preprocessing Properties", self.preprocessing_results)
        self._print_category_results("Training Configuration", self.training_results)
        self._print_category_results("Output Transforms", self.output_transform_results)

        # Print overall compliance
        print("=" * 80)
        print("SAS COMPLIANCE ASSESSMENT")
        print("=" * 80)

        compliance_score = (self.passed_tests / self.total_tests) * 100

        if compliance_score >= 95:
            compliance_level = "EXCELLENT"
        elif compliance_score >= 85:
            compliance_level = "GOOD"
        elif compliance_score >= 70:
            compliance_level = "MODERATE"
        else:
            compliance_level = "NEEDS IMPROVEMENT"

        print(f"Overall SAS Compliance: {compliance_score:.1f}% ({compliance_level})")
        print()

    def _print_category_results(self, category_name: str, results: dict):
        """Print results for a specific category."""
        if not results:
            return

        print(f"{category_name}:")
        print("-" * len(category_name))

        for test_name, result_data in results.items():
            status = result_data['result']
            details = result_data['details']

            # Format test name for readability
            display_name = test_name.replace('test_', '').replace('_', ' ').title()

            status_symbol = {
                'PASS': '✓',
                'FAIL': '✗',
                'ERROR': '!'
            }.get(status, '?')

            print(f"  {status_symbol} {display_name:<50} {status}")
            if details and status != 'PASS':
                print(f"    Details: {details}")

        # Category summary
        category_total = len(results)
        category_passed = sum(1 for r in results.values() if r['result'] == 'PASS')
        category_rate = (category_passed / category_total) * 100 if category_total > 0 else 0

        print(f"  Category Summary: {category_passed}/{category_total} passed ({category_rate:.1f}%)")
        print()


class PropertyTestRunner:
    """Custom test runner for property tests."""

    def __init__(self):
        self.result = PropertyTestResult()

    def run_all_tests(self) -> PropertyTestResult:
        """Run all property tests and return results."""
        start_time = time.time()

        print("Running SAS Autoencoder Properties Tests...")
        print("=" * 50)

        # Test categories and their corresponding test classes
        test_categories = [
            ('architecture', TestArchitectureProperties),
            ('preprocessing', TestPreprocessingProperties),
            ('training', TestTrainingProperties),
            ('output_transform', TestOutputTransformProperties),
        ]

        for category, test_class in test_categories:
            print(f"\nRunning {category.replace('_', ' ').title()} Tests...")
            self._run_test_category(category, test_class)

        self.result.execution_time = time.time() - start_time
        return self.result

    def _run_test_category(self, category: str, test_class):
        """Run tests for a specific category."""
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)

        # Run tests
        runner = unittest.TextTestRunner(stream=open('/dev/null', 'w'), verbosity=0)
        test_result = runner.run(suite)

        # Process results
        for test, error in test_result.errors:
            test_name = test._testMethodName
            error_msg = str(error).split('\n')[0] if error else ""
            self.result.add_result(category, test_name, 'ERROR', error_msg)
            print(f"  ERROR: {test_name}")

        for test, failure in test_result.failures:
            test_name = test._testMethodName
            failure_msg = str(failure).split('\n')[0] if failure else ""
            self.result.add_result(category, test_name, 'FAIL', failure_msg)
            print(f"  FAIL:  {test_name}")

        # Count successful tests
        total_tests = test_result.testsRun
        failed_tests = len(test_result.failures) + len(test_result.errors)
        passed_tests = total_tests - failed_tests

        # Add passed tests to results
        all_test_methods = [method for method in dir(test_class)
                           if method.startswith('test_')]

        failed_test_names = set()
        failed_test_names.update(test._testMethodName for test, _ in test_result.failures)
        failed_test_names.update(test._testMethodName for test, _ in test_result.errors)

        for test_method in all_test_methods:
            if test_method not in failed_test_names:
                self.result.add_result(category, test_method, 'PASS')
                print(f"  PASS:  {test_method}")

        print(f"  Category Result: {passed_tests}/{total_tests} passed")


def run_property_tests():
    """Main function to run all property tests."""
    try:
        runner = PropertyTestRunner()
        result = runner.run_all_tests()
        result.print_summary()

        # Return appropriate exit code
        return 0 if result.failed_tests == 0 and result.error_tests == 0 else 1

    except Exception as e:
        print(f"Error running property tests: {e}")
        return 2


if __name__ == '__main__':
    exit_code = run_property_tests()
    sys.exit(exit_code)