/**
 * Minimal stub of @earendil-works/pi-ai for extension integration tests.
 *
 * shop_extension.ts only needs `Type` (its runtime value) to declare tool
 * parameter schemas; everything else from the pi packages is type-only and is
 * erased before execution. Keeping this stub tiny makes it obvious the tests
 * exercise our code, not a mock's opinions.
 */

const Type = {
	Object: (properties, options) => ({ type: "object", properties, options }),
	String: (options) => ({ type: "string", options }),
};

module.exports = { Type };
